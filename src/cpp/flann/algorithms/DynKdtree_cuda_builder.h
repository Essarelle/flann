#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <flann/util/cutil_math.h>
#include <stdlib.h>
#include <flann/algorithms/kdtree_cuda_builder.h>
#include <flann/util/DeviceMatrix.h>

namespace flann
{
namespace cuda
{
namespace dyn_kd_tree_builder_detail
{

	struct set_addr3
	{
		DeviceMatrix<int>::iterator val_;
		const int* f_;

		int npoints_;
		__device__
			int operator()(int id)
		{
			int nf = f_[npoints_ - 1] + (val_[npoints_ - 1]);
			int f = f_[id];
			int t = id - f + nf;
			return val_[id] ? f : t;
		}
	};
	template<typename T>
	struct MovePointsToChildNodes
	{
		MovePointsToChildNodes(int* child1, kd_tree_builder_detail::SplitInfo* splits, 
			DeviceMatrix<T> points, DeviceMatrix<int> ownership, DeviceMatrix<int> leftright, int D)
			: child1_(child1), splits_(splits), points_(points), ownership_(ownership), leftright_(leftright), D_(D){}
		//  int dim;
		//  float threshold;
		int* child1_;
		kd_tree_builder_detail::SplitInfo* splits_;

		// coordinate values
		//float* x_, *y_, *z_;
		DeviceMatrix<T> points_;
		// owner indices -> which node does the point belong to?
		DeviceMatrix<int> ownership_;
		//int* ox_, *oy_, *oz_;
		// temp info: will be set to 1 of a point is moved to the right child node, 0 otherwise
		// (used later in the scan op to separate the points of the children into continuous ranges)
		//int* lrx_, *lry_, *lrz_;
		DeviceMatrix<int> leftright_;
		const int D_;
		__device__
			void operator()(const thrust::tuple<int, int&>& data)
		{
			int index = thrust::get<0>(data);

			int owner = ownership_[index][0]; // before a split, all points at the same position in the index array have the same owner
			int* point_ind = &thrust::get<1>(data);
			/*int point_ind1 = thrust::get<1>(data);
			int point_ind2 = thrust::get<2>(data);
			int point_ind3 = thrust::get<3>(data);*/
			int leftChild = child1_[owner];
			int split_dim;
			float* dim_val = (float*)malloc(D_*sizeof(float));
			kd_tree_builder_detail::SplitInfo split;
					
			for (int i = 0; i < D_; ++i)
			{
				leftright_[index][i] = 0;
			}
			// this element already belongs to a leaf node -> everything alright, no need to change anything
			if (leftChild == -1) 
			{
				return;
			}
			// otherwise: load split data, and assign this index to the new owner
			split = splits_[owner];
			split_dim = split.split_dim;
			for (int i = 0; i < D_; ++i)
			{
				float dim_val = points_[point_ind[i]][split_dim];
				ownership_[index][i] = leftChild + (dim_val > split.split_val);
				leftright_[index][i] = (dim_val > split.split_val);
			}
		} // operator()
	}; // MovePointsToChildNodes

	template<typename T>
	struct SetLeftAndRightAndAABB
	{
		int maxPoints;
		int nElements;

		kd_tree_builder_detail::SplitInfo* nodes;
		int* counts;
		int* labels;
		DeviceMatrix<T> aabbMin;
		DeviceMatrix<T> aabbMax;
		DeviceMatrix<T> points_;
		DeviceMatrix<int> indecies_;
		//const int* ix, *iy, *iz;

		__host__ __device__
			void operator()(int i)
		{
			int index = labels[i];
			int right;
			int left = counts[i];
			nodes[index].left = left;
			if (i < nElements - 1) 
			{
				right = counts[i + 1];
			}
			else 
			{ // index==nNodes
				right = maxPoints;
			}
			nodes[index].right = right;
			for (int i = 0; i < aabbMin.cols; ++i)
			{
				aabbMin[index][i] = points_[indecies_[left][i]][i];
				aabbMax[index][i] = points_[indecies_[right - 1][i]][i];
			}
		} // operator()
	}; // SetLeftAndRightAndAABB

	//! - decide whether a node has to be split
	//! if yes:
	//! - allocate child nodes
	//! - set split axis as axis of maximum aabb length
	template<typename T>
	struct SplitNodes
	{
		int maxPointsPerNode;
		int* node_count;
		int* nodes_allocated;
		int* out_of_space;
		int* child1_;
		int* parent_;
		kd_tree_builder_detail::SplitInfo* splits;
		int D;

		__device__
			void operator()(thrust::tuple<int&, int&, kd_tree_builder_detail::SplitInfo&, T&, T&, int> node) // float4: aabbMin, aabbMax
		{
			int& parent = thrust::get<0>(node);
			int& child1 = thrust::get<1>(node);
			kd_tree_builder_detail::SplitInfo& s = thrust::get<2>(node);
			const T* aabbMin = &thrust::get<3>(node);
			const T* aabbMax = &thrust::get<4>(node);
			int my_index = thrust::get<5>(node);
			bool split_node = false;
			// first, each thread block counts the number of nodes that it needs to allocate...
			__shared__ int block_nodes_to_allocate;
			if (threadIdx.x == 0)
				block_nodes_to_allocate = 0;
			__syncthreads();

			// don't split if all points are equal
			// (could lead to an infinite loop, and doesn't make any sense anyway)
			//bool all_points_in_node_are_equal=aabbMin.x == aabbMax.x && aabbMin.y==aabbMax.y && aabbMin.z==aabbMax.z;
			bool all_points_in_node_are_equal = true;
			for (int i = 0; i < D; ++i)
			{
				if (aabbMin[i] != aabbMax[i])
					all_points_in_node_are_equal = false;
			}
			int offset_to_global = 0;

			// maybe this could be replaced with a reduction...
			if ((child1 == -1) && (s.right - s.left > maxPointsPerNode) && !all_points_in_node_are_equal)  // leaf node
			{
				split_node = true;
				offset_to_global = atomicAdd(&block_nodes_to_allocate, 2);
			}

			__syncthreads();
			__shared__ int block_left;
			__shared__ bool enough_space;
			// ... then the first thread tries to allocate this many nodes...
			if (threadIdx.x == 0)
			{
				block_left = atomicAdd(node_count, block_nodes_to_allocate);
				enough_space = block_left + block_nodes_to_allocate < *nodes_allocated;
				// if it doesn't succeed, no nodes will be created by this block
				if (!enough_space)
				{
					atomicAdd(node_count, -block_nodes_to_allocate);
					*out_of_space = 1;
				}
			}

			__syncthreads();
			// this thread needs to split it's node && there was enough space for all the nodes
			// in this block.
			//(The whole "allocate-per-block-thing" is much faster than letting each element allocate
			// its space on its own, because shared memory atomics are A LOT faster than
			// global mem atomics!)
			if (split_node && enough_space) 
			{
				int left = block_left + offset_to_global;

				splits[left].left = s.left;
				splits[left].right = s.right;
				splits[left + 1].left = 0;
				splits[left + 1].right = 0;

				// split axis/position: middle of longest aabb extent
				
				int maxDim = 0;
				T maxDimLength = aabbMax[0] - aabbMin[0];

				for (int i = 1; i < D; i++)
				{
					T val = aabbMax[i] - aabbMin[i];
					if (val > maxDimLength)
					{
						maxDim = i;
						maxDimLength = val;
					}
				}
				s.split_dim = maxDim;
				s.split_val = (aabbMax[maxDim] + aabbMin[maxDim])*0.5f; 

				child1_[my_index] = left;
				splits[my_index] = s;

				parent_[left] = my_index;
				parent_[left + 1] = my_index;
				child1_[left] = -1;
				child1_[left + 1] = -1;
			}
		} // operator()
	}; // SplitNodes

	// this version is used for dynamically determined multi dimensional datasets
	template<typename T>
	class CudaKdTreeBuilder
	{
	public:
		CudaKdTreeBuilder(DeviceMatrix<T> points, int max_leaf_size) :
			max_leaf_size_(max_leaf_size_), points_(points)
		{
			int prealloc = points.rows / max_leaf_size_ * 16;
			allocation_info_.resize(3);
			allocation_info_[NodeCount] = 1;
			allocation_info_[NodesAllocated] = prealloc;
			allocation_info_[OutOfSpace] = 0;

			child1_ = new thrust::device_vector<int>(prealloc, -1);
			parent_ = new thrust::device_vector<int>(prealloc, -1);

			cuda::kd_tree_builder_detail::SplitInfo s;
			s.left = 0;
			s.right = 0;
			splits_ = new thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo>(prealloc, s);
			s.right = points.rows;
			(*splits_)[0] = s;

			d_aabb_min_ = new thrust::device_vector<T>(prealloc*points.cols);
			d_aabb_max_ = new thrust::device_vector<T>(prealloc*points.cols);
			aabb_min_ = flann::cuda::DeviceMatrix<T>(thrust::raw_pointer_cast(d_aabb_min_->data()), prealloc, points.cols);
			aabb_max_ = flann::cuda::DeviceMatrix<T>(thrust::raw_pointer_cast(d_aabb_max_->data()), prealloc, points.cols);

			d_index_ = new thrust::device_vector<int>(points_.rows*points_.cols);
			index_ = flann::cuda::DeviceMatrix<int>(thrust::raw_pointer_cast(d_index_->data()), points_.rows, points_.cols);

			d_owners_ = new thrust::device_vector<int>(points_.rows*points_.cols, 0);
			owners_ = flann::cuda::DeviceMatrix<int>(thrust::raw_pointer_cast(d_owners_->data()), points_.rows, points_.cols);

			d_leftright_ = new thrust::device_vector<int>(points_.rows * points_.cols, 0);
			leftright_ = flann::cuda::DeviceMatrix<int>(thrust::raw_pointer_cast(d_leftright_->data()), points_.rows, points_.cols);
					
			delete_node_info_ = false;
		} // CudaKdTreeBuilder

		void buildTree()
		{
			// Create GPU index arrays
			thrust::counting_iterator<int> it(0);
			thrust::copy(it, it + points_.rows, index_.begin());
			for (int i = 1; i < points_.cols; ++i)
			{
				thrust::copy(index_.begin(0), index_.end(0), index_.begin(i));
			}
			thrust::device_vector<float> tempv(points_.rows);
			for (int i = 0; i < points_.cols; ++i)
			{
				thrust::copy(points_.begin(i), points_.end(i), tempv.begin());
				thrust::sort_by_key(tempv.begin(), tempv.end(), index_.begin(i));
			}
			// Initialize max and min
			thrust::copy(thrust::device_pointer_cast(points_.ptr()), thrust::device_pointer_cast(points_.ptr() + points_.cols), thrust::device_pointer_cast(aabb_max_.ptr()));
			thrust::copy(thrust::device_pointer_cast(points_.ptr()), thrust::device_pointer_cast(points_.ptr() + points_.cols), thrust::device_pointer_cast(aabb_min_.ptr()));

			int last_node_count = 0;
			for (int i = 0;; i++)
			{
				SplitNodes<float> sn;
					sn.maxPointsPerNode = max_leaf_size_;
					sn.node_count = thrust::raw_pointer_cast(allocation_info_.data() + NodeCount);
					sn.nodes_allocated = thrust::raw_pointer_cast(allocation_info_.data() + NodesAllocated);
					sn.out_of_space = thrust::raw_pointer_cast(allocation_info_.data() + OutOfSpace);
					sn.child1_ = thrust::raw_pointer_cast(child1_->data());
					sn.parent_ = thrust::raw_pointer_cast(parent_->data());
					sn.splits = thrust::raw_pointer_cast(splits_->data());
					sn.D = points_.cols;

				thrust::counting_iterator<int> cit(0);
				thrust::for_each(
					thrust::make_zip_iterator(
						thrust::make_tuple(
							parent_->begin(),
							child1_->begin(),
							splits_->begin(),
							aabb_min_.begin(),
							aabb_max_.begin(),
							cit)),
					thrust::make_zip_iterator(
						thrust::make_tuple(
							parent_->begin() + last_node_count,
							child1_->begin() + last_node_count,
							splits_->begin() + last_node_count,
							aabb_min_.begin(0) + last_node_count, // This quite possibly needs to be last_node_count*dimensions, however aab_min_.begin() should point to the beginning of a row and increment by a row 
							aabb_max_.begin(0) + last_node_count,
							cit + last_node_count)),
					sn);
				// Get the allocation information from the run
				thrust::host_vector<int> alloc_info = allocation_info_;

				// no more nodes were split -> done
				if (last_node_count == alloc_info[NodeCount])
				{
					break;
				}

				last_node_count = alloc_info[NodeCount];

				// a node was un-splittable due to a lack of space
				if (alloc_info[OutOfSpace] == 1)
				{
					resize_node_vectors(alloc_info[NodesAllocated] * 2);
					alloc_info[OutOfSpace] = 0;
					alloc_info[NodesAllocated] *= 2;
					allocation_info_ = alloc_info;
				}

				MovePointsToChildNodes<T> sno( 
					thrust::raw_pointer_cast(child1_->data()),
					thrust::raw_pointer_cast(splits_->data()),
					points_, 
					owners_, 
					leftright_, 
					points_.cols);

				thrust::counting_iterator<int> ci0(0);
						
				thrust::for_each(
					thrust::make_zip_iterator(
						thrust::make_tuple(
							ci0,
							index_.begin(0))),
					thrust::make_zip_iterator(
						thrust::make_tuple(
							ci0 + points_.rows,
							index_.end(0))),
					sno);

				for (int i = 0; i < points_.cols; ++i)
				{
					separate_left_and_right_children(index_, owners_, *tmp_index_, *tmp_owners_, leftright_, i, i == 0);
					thrust::copy(tmp_index_->begin(), tmp_index_->end(), index_.begin(i));
					thrust::copy(tmp_owners_->begin(), tmp_owners_->end(), owners_.begin(i));
				}
				update_leftright_and_aabb(points_, index_, owners_, *splits_, aabb_min_, aabb_max_);
			}
		} // buildTree()

		void
			update_leftright_and_aabb(
				DeviceMatrix<T>& points,
				DeviceMatrix<int>& index,
				DeviceMatrix<int>& owners,
				thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo>& splits, 
				DeviceMatrix<T>& aabbMin, 
				DeviceMatrix<T>& aabbMax)
		{
			thrust::device_vector<int>* labelsUnique = tmp_owners_;
			thrust::device_vector<int>* countsUnique = tmp_index_;
			// assume: points of each node are continuous in the array

			// find which nodes are here, and where each node's points begin and end
			int unique_labels = thrust::unique_by_key_copy(
					owners.begin(0), 
					owners.end(0), 
					thrust::counting_iterator<int>(0), 
					labelsUnique->begin(), 
					countsUnique->begin()).first - labelsUnique->begin();

			// update the info
			SetLeftAndRightAndAABB<T> s;
				s.maxPoints = points.rows;
				s.nElements = unique_labels;
				s.nodes = thrust::raw_pointer_cast(splits.data());
				s.counts = thrust::raw_pointer_cast(countsUnique->data());
				s.labels = thrust::raw_pointer_cast(labelsUnique->data());
				s.points_ = points;
				s.indecies_ = index;
				s.aabbMin = aabbMin;
				s.aabbMax = aabbMax;

			thrust::counting_iterator<int> it(0);
			thrust::for_each(it, it + unique_labels, s);
		} // update_leftright_and_aabb


		//! Separates the left and right children of each node into continuous parts of the array.
		//! More specifically, it seperates children with even and odd node indices because nodes are always
		//! allocated in pairs -> child1==child2+1 -> child1 even and child2 odd, or vice-versa.
		//! Since the split operation is stable, this results in continuous partitions
		//! for all the single nodes.
		//! (basically the split primitive according to sengupta et al)
		//! about twice as fast as thrust::partition
		void separate_left_and_right_children(
			DeviceMatrix<int> key_in,
			DeviceMatrix<int> val_in,
			thrust::device_vector<int>& key_out, 
			thrust::device_vector<int>& val_out, 
			DeviceMatrix<int> left_right_marks,
			int Dim,
			bool scatter_val_out = true)
		{
			thrust::device_vector<int>* f_tmp = &val_out;
			thrust::device_vector<int>* addr_tmp = tmp_misc_;

			thrust::exclusive_scan(  left_right_marks.begin(Dim),left_right_marks.end(Dim), f_tmp->begin());
			set_addr3 sa;
			sa.val_ = left_right_marks.begin(Dim);
			sa.f_ = thrust::raw_pointer_cast(f_tmp->data());
			sa.npoints_ = key_in.rows;
			thrust::counting_iterator<int> it(0);
			thrust::transform(it, it + val_in.rows, addr_tmp->begin(), sa);

			thrust::scatter(key_in.begin(Dim), key_in.end(Dim), addr_tmp->begin(), key_out.begin());
			if (scatter_val_out) 
				thrust::scatter(val_in.begin(Dim), val_in.end(Dim), addr_tmp->begin(), val_out.begin());
		} // separate_left_and_right_children

		void resize_node_vectors(size_t new_size)
		{
			size_t add = new_size - child1_->size();
			child1_->insert(child1_->end(), add, -1);
			parent_->insert(parent_->end(), add, -1);
			cuda::kd_tree_builder_detail::SplitInfo s;
				s.left = 0;
				s.right = 0;
			splits_->insert(splits_->end(), add, s);
			float f;
			d_aabb_min_->insert(d_aabb_min_->end(), add*points_.cols, f);
			d_aabb_max_->insert(d_aabb_max_->end(), add*points_.cols, f);
			aabb_min_ = flann::cuda::DeviceMatrix<T>(thrust::raw_pointer_cast(d_aabb_min_->data()), add, points_.cols);
			aabb_max_ = flann::cuda::DeviceMatrix<T>(thrust::raw_pointer_cast(d_aabb_max_->data()), add, points_.cols);
		} // resize_node_vector

		template<class Distance>
		friend class KDTreeCuda3dIndex;
	protected:
		flann::cuda::DeviceMatrix<float> points_;
		// tree data, those are stored per-node

		//! left child of each node. (right child==left child + 1, due to the alloc mechanism)
		//! child1_[node]==-1 if node is a leaf node
		thrust::device_vector<int>* child1_;
		//! parent node of each node
		thrust::device_vector<int>* parent_;
		//! split info (dim/value or left/right pointers)
		thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo>* splits_;
		//! min aabb value of each node
		thrust::device_vector<T>* d_aabb_min_;
		flann::cuda::DeviceMatrix<T> aabb_min_;
		//! max aabb value of each node
		thrust::device_vector<T>* d_aabb_max_;
		flann::cuda::DeviceMatrix<T> aabb_max_;

		enum AllocationInfo
		{
			NodeCount = 0,
			NodesAllocated = 1,
			OutOfSpace = 2
		};
		thrust::device_vector<int> allocation_info_;

		int max_leaf_size_;

		// coordinate values of the points
		//thrust::device_vector<float>* points_x_, *points_y_, *points_z_;
		//thrust::device_vector<float>* d_sorted_points_;
		//flann::Matrix<float> sorted_points_;
		// indices
		//thrust::device_vector<int>* index_x_, *index_y_, *index_z_;
		thrust::device_vector<int>* d_index_;
		flann::cuda::DeviceMatrix<int> index_;
		// owner node
		//thrust::device_vector<int>* owners_x_, *owners_y_, *owners_z_;
		thrust::device_vector<int>* d_owners_;
		flann::cuda::DeviceMatrix<int> owners_;
		// contains info about whether a point was partitioned to the left or right child after a split
		//thrust::device_vector<int>* leftright_x_, *leftright_y_, *leftright_z_;
		thrust::device_vector<int>* d_leftright_;
		flann::cuda::DeviceMatrix<int> leftright_;
		thrust::device_vector<int>* tmp_index_, *tmp_owners_, *tmp_misc_;
		bool delete_node_info_;

	}; // CudaKdTreeBuilder
}
}
}

