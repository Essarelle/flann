#pragma once
#include <flann/algorithms/DynKdtree_cuda_builder.cuh>
#include <flann/util/DeviceMatrix.h>
#include <flann/algorithms/kdtree_cuda_3d_index.h> // For params class
#include <flann/flann.hpp>
#include <flann/algorithms/dist.h>
#include <flann/util/cuda/result_set.h>
namespace flann
{
	namespace DynKdTreeCudaPrivate
	{
		struct map_indices
		{
			const int* v_;

			map_indices(const int* v) : v_(v) {
			}

			__host__ __device__
				float operator() (const int&i) const
			{
				if (i >= 0) return v_[i];
				else return i;
			}
		};
		template<typename T, typename GPUResultSet, typename Distance >
		__device__
			void searchNeighbors(const cuda::kd_tree_builder_detail::SplitInfo* splits,
			const int* child1,
			const int* parent,
			cuda::DeviceMatrix<T>& aabbLow,
			cuda::DeviceMatrix<T>& aabbHigh,
			cuda::DeviceMatrix<T>& elements,
			T* q,
			GPUResultSet& result, 
			const Distance& distance = Distance())
		{
			const int D = elements.cols;
			bool backtrack = false;
			int lastNode = -1;
			int current = 0;

			cuda::kd_tree_builder_detail::SplitInfo split;
			while (true) 
			{
				if (current == -1) 
					break;
				split = splits[current];

				T diff1;
				diff1 = q[split.split_dim] - split.split_val;
				

				// children are next to each other: leftChild+1 == rightChild
				int leftChild = child1[current];
				int bestChild = leftChild;
				int otherChild = leftChild;

				if (diff1<0) 
				{
					otherChild++;
				}
				else 
				{
					bestChild++;
				}

				if (!backtrack) 
				{
					/* If this is a leaf node, then do check and return. */
					if (leftChild == -1) 
					{
						for (int i = split.left; i<split.right; ++i) 
						{
							T dist = distance.dist(thrust::raw_pointer_cast(elements[i]), q, D);
							result.insert(i, dist);
						}
						backtrack = true;
						lastNode = current;
						current = parent[current];
					}
					else 
					{ // go to closer child node
						lastNode = current;
						current = bestChild;
					}
				}
				else 
				{ // continue moving back up the tree or visit far node?
					// minimum possible distance between query point and a point inside the AABB
					T mindistsq = 0;

					T* aabbMin = thrust::raw_pointer_cast(aabbLow[otherChild]);
					T* aabbMax = thrust::raw_pointer_cast(aabbHigh[otherChild]);

					for (int d = 0; d < D; ++d)
					{
						if(q[d] < aabbMin[d])
							mindistsq += distance.axisDist(q[d], aabbMin[d]);
						else
						{
							if (q[d] > aabbMax[d])
								mindistsq += distance.axisDist(q[d], aabbMax[d]);
						}
					}

					//  the far node was NOT the last node (== not visited yet) AND there could be a closer point in it
					if ((lastNode == bestChild) && 
						mindistsq <= result.worstDist()) 
					{
						lastNode = current;
						current = otherChild;
						backtrack = false;
					}
					else 
					{
						lastNode = current;
						current = parent[current];
					}
				}
			}
		}
		
		template<typename T, typename GPUResultSet, typename Distance >
		__global__
			void nearestKernel(const cuda::kd_tree_builder_detail::SplitInfo* splits,
			const int* child1,
			const int* parent,
			cuda::DeviceMatrix<T> aabbMin,
			cuda::DeviceMatrix<T> aabbMax,
			cuda::DeviceMatrix<T> elements,
			cuda::DeviceMatrix<T> query,
			cuda::DeviceMatrix<int> resultIdx,
			cuda::DeviceMatrix<T> resultDist,
			GPUResultSet result, Distance dist = Distance())
		{
			typedef T DistanceType;
			typedef T ElementType;

			size_t tid = blockDim.x*blockIdx.x + threadIdx.x;

			if (tid >= query.rows)
				return;

			result.setResultLocation(thrust::raw_pointer_cast(resultDist.ptr()),
				thrust::raw_pointer_cast(resultIdx.ptr()), tid, resultDist.stride / sizeof(T));

			searchNeighbors(splits, child1, parent, aabbMin, aabbMax, elements, thrust::raw_pointer_cast(query[tid]), result, dist);

			result.finish();
		}
		struct isNotMinusOne
		{
			__host__ __device__
				bool operator() (int i){
				return i != -1;
			}
		};
	}

	

	template<typename T>
	struct DynDistL2
	{
		static T __host__ __device__ axisDist(T a, T b)
		{
			return (a - b)*(a - b);
		}
		static T __host__ __device__ dist(T* a, T* b, size_t len)
		{
			T ret = 0;
			for (size_t i = 0; i < len; ++i)
			{
				ret = (a[i] - b[i])*(a[i] - b[i]);
			}
			return ret;
		}
	};
	template<typename T>
	struct DynDistL1
	{
		static T __host__ __device__ axisDist(T a, T b)
		{
			return fabs(a - b);
		}
		static T __host__ __device__ dist(T* a, T* b, size_t len)
		{
			T ret = 0;
			for (size_t i = 0; i < len; ++i)
			{
				ret = fabs(a[i] - b[i]);
			}
			return ret;
		}
	};
	template<typename T>
	struct DynDistHamming
	{
		static T __host__ __device__ axisDist(T a, T b)
		{
			return a != b;
		}
		static T __host__ __device__ dist(T* a, T* b, size_t len)
		{
			T ret = 0;
			for (size_t i = 0; i < len; ++i)
			{
				ret += a[i] != b[i];
			}
			return ret;
		}
	};

	template<class Distance>
	struct DynGpuDist	{	};

	template<typename T> 
	struct DynGpuDist< L2<T> >
	{
		typedef DynDistL2<T> type;
	};
	
	template<typename T>
	struct DynGpuDist< L1<T> >
	{
		typedef DynDistL1<T> type;
	};

	template<typename T>
	struct DynGpuDist< HammingPopcnt<T> >
	{
		typedef DynDistHamming<T> type;
	};


	template<typename T>
	struct DynGpuHelper
	{
		thrust::device_vector< cuda::kd_tree_builder_detail::SplitInfo >* gpu_splits_;
		thrust::device_vector< int >* gpu_parent_;
		thrust::device_vector< int >* gpu_child1_;
		thrust::device_vector<T> d_gpu_points_;
		flann::cuda::DeviceMatrix<T> gpu_points_;
		flann::cuda::DeviceMatrix<T> gpu_aabb_min_;
		flann::cuda::DeviceMatrix<T> gpu_aabb_max_;
		thrust::device_vector<int>* gpu_vind_;

		DynGpuHelper() : gpu_splits_(0), gpu_parent_(0), gpu_child1_(0), gpu_vind_(0){}
		~DynGpuHelper()
		{
			delete gpu_splits_;
			gpu_splits_ = 0;
			delete gpu_parent_;
			gpu_parent_ = 0;
			delete gpu_child1_;
			gpu_child1_ = 0;
			//delete gpu_aabb_max_;
			//gpu_aabb_max_ = 0;
			//delete gpu_aabb_min_;
			//gpu_aabb_min_ = 0;
			delete gpu_vind_;
			gpu_vind_ = 0;

			//delete gpu_points_;
			//gpu_points_ = 0;
		}
	};

	template<typename Distance>
	class KDTreeCudaIndex: public NNIndex<Distance>
	{
	public:
		typedef typename Distance::ElementType ElementType;
		typedef typename Distance::ResultType DistanceType;
		typedef NNIndex<Distance> BaseClass;

		int visited_leafs;
		
		typedef bool needs_kdtree_distance;

		KDTreeCudaIndex(
			const cuda::DeviceMatrix<ElementType>& inputData,
			const IndexParams& params = KDTreeCuda3dIndexParams(),
			Distance d = Distance()
			) :
			BaseClass(params, d),
				dataset_(inputData),
				leaf_count_(0),
				visited_leafs(0),
				node_count_(0),
				current_node_count_(0)
		{
			size_ = dataset_.rows;
			dim_ = dataset_.cols;

			int dim_param = get_param(params, "dim", -1);
			if (dim_param>0) dim_ = dim_param;
			leaf_max_size_ = get_param(params, "leaf_max_size", 10);
			gpu_helper_ = 0;
		}
		~KDTreeCudaIndex()
		{
			delete gpu_helper_;
		}

		virtual void buildIndex()
		{
			vind_.resize(size_);
			for (size_t i = 0; i < size_; ++i)
			{
				vind_[i] = i;
			}
			leaf_count_ = 0;
			node_count_ = 0;
			
			uploadTreeToGpu();
		}
		
		void knnSearchGpu(const cuda::DeviceMatrix<ElementType>& queries, 
			cuda::DeviceMatrix<int>& indices, 
			cuda::DeviceMatrix<DistanceType>& dists, 
			size_t knn, const SearchParams& params, cudaStream_t stream) const
		{
			assert(indices.rows >= queries.rows);
			assert(dists.rows >= queries.rows);
			assert(int(indices.cols) >= knn);
			assert(dists.cols == indices.cols && dists.stride == indices.stride);
			int threadsPerBlock = 128;
			int blocksPerGrid = (queries.rows + threadsPerBlock - 1) / threadsPerBlock;
			float epsError = 1 + params.eps;
			bool sorted = params.sorted;
			bool use_heap = params.use_heap;
			typename DynGpuDist<Distance>::type distance;
			if (knn == 1)
			{
				DynKdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
					gpu_helper_->gpu_aabb_min_,
					gpu_helper_->gpu_aabb_max_,
					gpu_helper_->gpu_points_,
					queries,
					indices,
					dists,
					flann::cuda::SingleResultSet<float>(epsError), distance);
			}
			else
			{
				if (use_heap)
				{
					DynKdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
						thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
						thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
						thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
						gpu_helper_->gpu_aabb_min_,
						gpu_helper_->gpu_aabb_max_,
						gpu_helper_->gpu_points_,
						queries,
						indices,
						dists, 
						flann::cuda::KnnResultSet<float, true>(knn, sorted, epsError), 
						distance);
				}
				else
				{
					DynKdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
						thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
						thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
						thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
						gpu_helper_->gpu_aabb_min_,
						gpu_helper_->gpu_aabb_max_,
						gpu_helper_->gpu_points_,
						queries,
						indices,
						dists, flann::cuda::KnnResultSet<float, false>(knn, sorted, epsError),
						distance
						);
				}
			}
			thrust::transform(thrust::system::cuda::par.on(stream), indices.ptr(), indices.ptr() + knn*queries.rows, indices.ptr(), 
				DynKdTreeCudaPrivate::map_indices(thrust::raw_pointer_cast(&((*gpu_helper_->gpu_vind_))[0])));
			if (stream == NULL)
				cudaDeviceSynchronize();
		}

		int radiusSearchGpu(const cuda::DeviceMatrix<ElementType>& queries, cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists, float radius, const SearchParams& params, cudaStream_t stream = NULL) const
		{
			int max_neighbors = params.max_neighbors;
			if (max_neighbors<0) 
				max_neighbors = indices.cols;
			assert(indices.rows >= queries.rows);
			assert(dists.rows >= queries.rows || max_neighbors == 0);
			assert(indices.stride == dists.stride || max_neighbors == 0);
			assert(indices.cols == indices.stride / sizeof(int));
			assert(dists.rows >= queries.rows || max_neighbors == 0);

			bool sorted = params.sorted;
			float epsError = 1 + params.eps;
			bool use_heap = params.use_heap;

			typename DynGpuDist<Distance>::type distance;
			int threadsPerBlock = 128;
			int blocksPerGrid = (queries.rows + threadsPerBlock - 1) / threadsPerBlock;

			if (max_neighbors == 0) 
			{
				//thrust::device_vector<int> indicesDev(queries.rows* indices.stride);
				DynKdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
					gpu_helper_->gpu_aabb_min_,
					gpu_helper_->gpu_aabb_max_,
					gpu_helper_->gpu_points_,
					queries,
					indices,
					dists,
					flann::cuda::CountingRadiusResultSet<float>(radius, -1),
					distance
					);
				//thrust::copy(thrust::system::cuda::par.on(stream), indicesDev.begin(), indicesDev.end(), indices.ptr());
				if (stream == NULL)
				{
					cudaDeviceSynchronize();
					return thrust::reduce(indices.begin(), indices.end());
				}
				return 0;
			}
			if (use_heap) {
				DynKdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
					gpu_helper_->gpu_aabb_min_,
					gpu_helper_->gpu_aabb_max_,
					gpu_helper_->gpu_points_,
					queries,
					indices,
					dists,
					flann::cuda::KnnRadiusResultSet<float, true>(max_neighbors, sorted, epsError, radius), 
					distance);
			}
			else {
				DynKdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
					thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
					gpu_helper_->gpu_aabb_min_,
					gpu_helper_->gpu_aabb_max_,
					gpu_helper_->gpu_points_,
					queries,
					indices,
					dists, 
					flann::cuda::KnnRadiusResultSet<float, false>(max_neighbors, sorted, epsError, radius), 
					distance);
			}

			thrust::transform(thrust::system::cuda::par.on(stream), indices.begin(), indices.end(), indices.begin(), 
				DynKdTreeCudaPrivate::map_indices(thrust::raw_pointer_cast(&((*gpu_helper_->gpu_vind_))[0])));
			if (stream == NULL)
			{
				cudaDeviceSynchronize();
				return thrust::count_if(indices.begin(), indices.end(), DynKdTreeCudaPrivate::isNotMinusOne());
			}
			return queries.rows * max_neighbors;

			return 0;
		}
		flann_algorithm_t getType() const
		{
			return FLANN_INDEX_KDTREE_SINGLE;
		}


		void removePoint(size_t index)
		{
			throw FLANNException("removePoint not implemented for this index type!");
		}

		ElementType* getPoint(size_t id)
		{
			return thrust::raw_pointer_cast(dataset_[id]);
		}

		void saveIndex(FILE* stream)
		{
			throw FLANNException("Index saving not implemented!");

		}


		void loadIndex(FILE* stream)
		{
			throw FLANNException("Index loading not implemented!");
		}
		int usedMemory() const
		{
			//         return tree_.size()*sizeof(Node)+dataset_.rows*sizeof(int);  // pool memory and vind array memory
			return 0;
		}
		BaseClass* clone() const
		{
			return nullptr;
		}
		virtual void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
		{
		}
	protected:
		void buildIndexImpl()
		{
			/* nothing to do here */
		}

		void freeIndex()
		{
			/* nothing to do here */
		}

	private:
		

		virtual void uploadTreeToGpu()
		{
			delete gpu_helper_;
			gpu_helper_ = new DynGpuHelper<ElementType>;
			gpu_helper_->d_gpu_points_.resize(dataset_.rows * dataset_.cols);
			gpu_helper_->gpu_points_ = flann::cuda::DeviceMatrix<ElementType>(thrust::raw_pointer_cast(gpu_helper_->d_gpu_points_.data()),
				dataset_.rows, dataset_.cols);
			::flann::cuda::dyn_kd_tree_builder_detail::CudaKdTreeBuilder<ElementType> builder(dataset_, leaf_max_size_);

			builder.buildTree();
			
			gpu_helper_->gpu_splits_ = builder.splits_;
			gpu_helper_->gpu_aabb_max_ = builder.aabb_max_;
			gpu_helper_->gpu_aabb_min_ = builder.aabb_min_;
			gpu_helper_->gpu_child1_ = builder.child1_;
			gpu_helper_->gpu_parent_ = builder.parent_;
			
			if (gpu_helper_->gpu_vind_ == nullptr)
				gpu_helper_->gpu_vind_ = new thrust::device_vector<int>();

			gpu_helper_->gpu_vind_->insert(gpu_helper_->gpu_vind_->begin(), builder.index_.begin(0), builder.index_.end(0));
			
			for (int i = 0; i < dataset_.cols; ++i)
			{
				thrust::gather(builder.index_.begin(0), builder.index_.end(0), dataset_.begin(i), gpu_helper_->gpu_points_.begin(i));
			}
			

			//thrust::gather(builder.index_.begin(0), builder.index_.end(0), )
			//gpu_helper_->gpu_vind_ = builder.index_;
		}
		
		friend class DynGpuHelper<ElementType>;
		DynGpuHelper<ElementType>* gpu_helper_;

		flann::cuda::DeviceMatrix<ElementType> dataset_;

		int leaf_max_size_;

		int leaf_count_;
		int node_count_;
		//! used by convertTreeToGpuFormat
		int current_node_count_;
		std::vector<int> vind_;

		flann::cuda::DeviceMatrix<ElementType> data_;

		size_t dim_;

		USING_BASECLASS_SYMBOLS

	};// KDTreeCudaIndex

	template<typename Distance>
	class DynGpuIndex
	{
	public:
		typedef typename Distance::ElementType ElementType;
		typedef typename Distance::ResultType DistanceType;
		typedef KDTreeCudaIndex<Distance> IndexType;
		DynGpuIndex(const IndexParams& params, Distance distance = Distance())
		{
			index_params_ = params;
			//nnIndex = new KDTreeCudaIndex<Distance>()


		}
		DynGpuIndex(cuda::DeviceMatrix<ElementType> features, const IndexParams& params, Distance distance = Distance())
		{
			index_params_ = params;
			nnIndex_ = new KDTreeCudaIndex<Distance>(features, params, distance);
			//nnIndex_->buildIndex();
		}
		~DynGpuIndex()
		{
			delete nnIndex_;
		}
		virtual void buildIndex()
		{
			nnIndex_->buildIndex();
		}
		virtual int knnSearch(cuda::DeviceMatrix<ElementType> queries,
			cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists,
			size_t knn,
			const SearchParams& params,
			cudaStream_t stream = NULL) const
		{
			nnIndex_->knnSearchGpu(queries, indices, dists, knn, params, stream);
			return 0;
		}
		virtual int radiusSearch(cuda::DeviceMatrix<ElementType> queries,
			cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists,
			float radius,
			const SearchParams& params,
			cudaStream_t stream = NULL) const
		{
			return nnIndex_->radiusSearchGpu(queries, indices, dists, radius, params, stream);
		}
	private:
		KDTreeCudaIndex<Distance>* nnIndex_;
		IndexParams index_params_;

	};
}