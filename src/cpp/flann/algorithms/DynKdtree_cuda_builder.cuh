#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <flann/util/cutil_math.h>
#include <stdlib.h>
#include <flann/algorithms/kdtree_cuda_builder.cuh>
#include <flann/util/DeviceMatrix.h>
#include <memory>
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
		DeviceMatrix<T> points_;
		// owner indices -> which node does the point belong to?
		DeviceMatrix<int> ownership_;
		// temp info: will be set to 1 of a point is moved to the right child node, 0 otherwise
		// (used later in the scan op to separate the points of the children into continuous ranges)
		DeviceMatrix<int> leftright_;
		const int D_;
		__device__
			void operator()(const thrust::tuple<int, int&>& data)
		{
			int index = thrust::get<0>(data);

			int owner = ownership_[index][0]; // before a split, all points at the same position in the index array have the same owner
			int* point_ind = &thrust::get<1>(data);
			int leftChild = child1_[owner];
			int split_dim;
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
			if (split_dim >= D_)
			{
				printf("invalid split dim %d", split_dim);
			}
			else
			{
				for (int i = 0; i < D_; ++i)
				{
					T dim_val = points_[point_ind[i]][split_dim];
					ownership_[index][i] = leftChild + (dim_val > split.split_val);
					leftright_[index][i] = (dim_val > split.split_val);
				}
			}
			

		} // operator()
	}; // MovePointsToChildNodes

	template<typename T>
	struct SetLeftAndRightAndAABB
	{
		int maxPoints;
		int nElements;
		int splitSize;

		kd_tree_builder_detail::SplitInfo* nodes;
		int* counts;
		int* labels;
		DeviceMatrix<T> aabbMin;
		DeviceMatrix<T> aabbMax;
		DeviceMatrix<T> points_;
		DeviceMatrix<int> indecies_;

		__host__ __device__
			void operator()(int i)
		{
			int index = labels[i];
			if (index > splitSize)
			{
				printf("Index > splitSize");
			}
			int right;
			int left = counts[i];
			nodes[index].left = left;
			if (i < nElements - 1) 
			{
				right = counts[i + 1];
			}
			else 
			{
				right = maxPoints;
			}
			nodes[index].right = right;
			for (int d = 0; d < aabbMin.cols; ++d)
			{
				int leftIndex = indecies_[left][d];
				int rightIndex = indecies_[right - 1][d];

				aabbMin[index][d] = points_[leftIndex][d];
				aabbMax[index][d] = points_[rightIndex][d];
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
			void operator()(thrust::tuple<int&, int&, kd_tree_builder_detail::SplitInfo&, T&, T&, int> node)
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
				if (maxDim > 3)
				{
					printf("Max dim > 3");
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
			max_leaf_size_(max_leaf_size), points_(points)
		{
			int prealloc = points.rows / max_leaf_size_ * 16;
			allocation_info_.resize(3);
			allocation_info_[NodeCount] = 1;
			allocation_info_[NodesAllocated] = prealloc;
			allocation_info_[OutOfSpace] = 0;

			child1_.reset(new thrust::device_vector<int>(prealloc, -1));
			parent_.reset(new thrust::device_vector<int>(prealloc, -1));

			cuda::kd_tree_builder_detail::SplitInfo s;
				s.left = 0;
				s.right = 0;
			splits_.reset(new thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo>(prealloc, s));
			s.right = points.rows;
			(*splits_)[0] = s;

			d_aabb_min_.reset(new thrust::device_vector<T>(prealloc*points.cols));
			d_aabb_max_.reset(new thrust::device_vector<T>(prealloc*points.cols));
			aabb_min_ = flann::cuda::DeviceMatrix<T>(thrust::raw_pointer_cast(d_aabb_min_->data()), prealloc, points.cols);
			aabb_max_ = flann::cuda::DeviceMatrix<T>(thrust::raw_pointer_cast(d_aabb_max_->data()), prealloc, points.cols);

			d_index_.reset(new thrust::device_vector<int>(points_.rows*points_.cols));
			index_ = flann::cuda::DeviceMatrix<int>(thrust::raw_pointer_cast(d_index_->data()), points_.rows, points_.cols);

			d_owners_.reset(new thrust::device_vector<int>(points_.rows*points_.cols, 0));
			owners_ = flann::cuda::DeviceMatrix<int>(thrust::raw_pointer_cast(d_owners_->data()), points_.rows, points_.cols);

			d_leftright_.reset(new thrust::device_vector<int>(points_.rows * points_.cols, 0));
			leftright_ = flann::cuda::DeviceMatrix<int>(thrust::raw_pointer_cast(d_leftright_->data()), points_.rows, points_.cols);

			tmp_index_.reset(new thrust::device_vector<int>(points_.rows));
			tmp_owners_.reset(new thrust::device_vector<int>(points_.rows));
			tmp_misc_.reset(new thrust::device_vector<int>(points_.rows));
					
			delete_node_info_ = false;
		} // CudaKdTreeBuilder

		void buildTree()
		{
			// Create GPU index arrays
			thrust::counting_iterator<int> it(0);
			//thrust::copy(it, it + points_.rows, index_.begin());
			thrust::sequence(index_.begin(0), index_.end(0));
			for (int d = 1; d < points_.cols; ++d)
			{
				thrust::copy(index_.begin(0), index_.end(0), index_.begin(d));
			}
			thrust::device_vector<float> tempv(points_.rows);
			for (int d = 0; d < points_.cols; ++d)
			{
				thrust::copy(points_.begin(d), points_.end(d), tempv.begin());
				thrust::sort_by_key(tempv.begin(), tempv.end(), index_.begin(d));
			}
			// Initialize max and min
			for (int d = 0; d < points_.cols; ++d)
			{
				int idxMin = (*d_index_)[d];
				int idxMax = (*d_index_)[(points_.rows - 1) * points_.cols + d];
				cudaMemcpy((void*)(thrust::raw_pointer_cast((*d_aabb_max_).data()) + d), (void*)(thrust::raw_pointer_cast(points_[idxMin]) + d), sizeof(T), cudaMemcpyDeviceToDevice);
				cudaMemcpy((void*)(thrust::raw_pointer_cast((*d_aabb_min_).data()) + d), (void*)(thrust::raw_pointer_cast(points_[idxMax]) + d), sizeof(T), cudaMemcpyDeviceToDevice);
			}

			int last_node_count = 0;
			for (int i = 0;; i++)
			{
				/*thrust::host_vector<cuda::kd_tree_builder_detail::SplitInfo> h_splits = *splits_;
				int count = 0;
				int index = 0;
				for (auto itr = h_splits.begin(); itr != h_splits.end(); ++itr, ++index)
				{
					if ((*itr).split_dim > 3 && count < 3)
						std::cout << "0-Iteration: " << i << " Invalid split " << ++count << " at index: " << index << std::endl;

				}*/


				SplitNodes<T> sn;
					sn.maxPointsPerNode = max_leaf_size_;
					sn.node_count = thrust::raw_pointer_cast(&allocation_info_[NodeCount]);
					sn.nodes_allocated = thrust::raw_pointer_cast(&allocation_info_[NodesAllocated]);
					sn.out_of_space = thrust::raw_pointer_cast(&allocation_info_[OutOfSpace]);
					sn.child1_ = thrust::raw_pointer_cast(&(*child1_)[0]);
					sn.parent_ = thrust::raw_pointer_cast(&(*parent_)[0]);
					sn.splits = thrust::raw_pointer_cast(&(*splits_)[0]);
					sn.D = points_.cols;
				
				thrust::counting_iterator<int> cit(0);
				thrust::for_each(
					thrust::make_zip_iterator(
						thrust::make_tuple(
							parent_->begin(),
							child1_->begin(),
							splits_->begin(),
							aabb_min_.begin(0),
							aabb_max_.begin(0),
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
				/*h_splits = *splits_;
				count = 0;
				index = 0;
				for (auto itr = h_splits.begin(); itr != h_splits.end(); ++itr, ++index)
				{
					if ((*itr).split_dim > 3 && count < 3)
						std::cout << "1-Iteration: " << i << " Invalid split " << ++count << " at index: " << index << std::endl;

				}*/
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
				/*h_splits = *splits_;
				count = 0;
				index = 0;
				for (auto itr = h_splits.begin(); itr != h_splits.end(); ++itr, ++index)
				{
					if ((*itr).split_dim > 3 && count < 3)
						std::cout << "2-Iteration: " << i << " Invalid split " << ++count << " at index: " << index << std::endl;

				}*/
				
				for (int d = 0; d < points_.cols; ++d)
				{
					separate_left_and_right_children(index_, owners_, *tmp_index_, *tmp_owners_, leftright_, d, d == 0);
					thrust::copy(tmp_index_->begin(), tmp_index_->end(), index_.begin(d));
					thrust::copy(tmp_owners_->begin(), tmp_owners_->end(), owners_.begin(d));
				}
				/*h_splits = *splits_;
				count = 0;
				index = 0;
				for (auto itr = h_splits.begin(); itr != h_splits.end(); ++itr, ++index)
				{
					if ((*itr).split_dim > 3 && count < 3)
						std::cout << "3-Iteration: " << i << " Invalid split " << ++count << " at index: " << index << std::endl;

				}*/
				update_leftright_and_aabb(points_, index_, owners_, *splits_, aabb_min_, aabb_max_);
				/*h_splits = *splits_;
				count = 0;
				index = 0;
				for (auto itr = h_splits.begin(); itr != h_splits.end(); ++itr, ++index)
				{
					if ((*itr).split_dim > 3 && count < 3)
						std::cout << "4-Iteration: " << i << " Invalid split " << ++count << " at index: " << index << std::endl;

				}*/
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
			thrust::device_vector<int>* labelsUnique = tmp_owners_.get();
			thrust::device_vector<int>* countsUnique = tmp_index_.get();
			// assume: points of each node are continuous in the array

			/*cv::cuda::GpuMat d_labels(labelsUnique->size(), 1, CV_32S, thrust::raw_pointer_cast(labelsUnique->data()));
			cv::cuda::GpuMat d_counts(countsUnique->size(), 1, CV_32S, thrust::raw_pointer_cast(countsUnique->data()));
			cv::cuda::GpuMat d_owners(owners.rows, owners.cols, CV_32S, (void*)thrust::raw_pointer_cast(owners.ptr()), owners.stride);
			cv::cuda::GpuMat d_points(points.rows, points.cols, CV_32F, (void*)thrust::raw_pointer_cast(points.ptr()), points.stride);
			cv::cuda::GpuMat d_splits(splits.size(), 2, CV_32S, (void*)thrust::raw_pointer_cast(splits.data()), sizeof(cuda::kd_tree_builder_detail::SplitInfo));
			cv::cuda::GpuMat d_aabbMin(aabbMin.rows, aabbMin.cols, CV_32F, (void*)thrust::raw_pointer_cast(aabbMin.ptr()), aabbMin.stride);
			cv::cuda::GpuMat d_aabbMax(aabbMax.rows, aabbMax.cols, CV_32F, (void*)thrust::raw_pointer_cast(aabbMax.ptr()), aabbMax.stride);


			cv::Mat h_labels(d_labels);
			cv::Mat h_counts(d_counts);
			cv::Mat h_points(d_points);
			cv::Mat h_splits(d_splits);
			cv::Mat h_owners(d_owners);
			cv::Mat h_aabbMin(d_aabbMin);
			cv::Mat h_aabbMax(d_aabbMax);*/

			// find which nodes are here, and where each node's points begin and end
			int unique_labels = thrust::unique_by_key_copy(
					owners.begin(0),  // Effectively owners_x.begin()
					owners.end(0), 
					thrust::counting_iterator<int>(0), 
					labelsUnique->begin(), 
					countsUnique->begin()).first - labelsUnique->begin();

			// update the info
			SetLeftAndRightAndAABB<T> s;
				s.maxPoints = points.rows;
				s.nElements = unique_labels;
				s.nodes = thrust::raw_pointer_cast(&(splits[0]));
				s.counts = thrust::raw_pointer_cast(&((*countsUnique)[0]));
				s.labels = thrust::raw_pointer_cast(&((*labelsUnique)[0]));
				s.points_ = points;
				s.indecies_ = index;
				s.aabbMin = aabbMin;
				s.aabbMax = aabbMax;
				s.splitSize = splits.size();

			thrust::counting_iterator<int> it(0);
			//std::cout << "Unique labels: " << unique_labels << std::endl;
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
			thrust::device_vector<int>* addr_tmp = tmp_misc_.get();

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

		flann::cuda::DeviceMatrix<T> points_;
		// tree data, those are stored per-node

		//! left child of each node. (right child==left child + 1, due to the alloc mechanism)
		//! child1_[node]==-1 if node is a leaf node
		std::shared_ptr<thrust::device_vector<int>> child1_;
		//! parent node of each node
		std::shared_ptr<thrust::device_vector<int>> parent_;
		//! split info (dim/value or left/right pointers)
		std::shared_ptr<thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo>> splits_;
		//! min aabb value of each node
		std::shared_ptr<thrust::device_vector<T>> d_aabb_min_;
		flann::cuda::DeviceMatrix<T> aabb_min_;
		//! max aabb value of each node
		std::shared_ptr<thrust::device_vector<T>> d_aabb_max_;
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
		//flann::Matrix<float> sorted_points_;
		// indices
		std::shared_ptr<thrust::device_vector<int>> d_index_;
		flann::cuda::DeviceMatrix<int> index_;
		// owner node
		std::shared_ptr<thrust::device_vector<int>> d_owners_;
		flann::cuda::DeviceMatrix<int> owners_;
		// contains info about whether a point was partitioned to the left or right child after a split
		std::shared_ptr<thrust::device_vector<int>> d_leftright_;
		flann::cuda::DeviceMatrix<int> leftright_;
		std::shared_ptr<thrust::device_vector<int>> tmp_index_, tmp_owners_, tmp_misc_;
		bool delete_node_info_;

	}; // CudaKdTreeBuilder
}
}
}

