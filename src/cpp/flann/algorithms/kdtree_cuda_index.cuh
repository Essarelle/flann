#pragma once
#include <flann/algorithms/DynKdtree_cuda_builder.cuh>
#include <flann/util/DeviceMatrix.h>
#include <flann/algorithms/kdtree_cuda_3d_index.h> // For params class
#include <flann/flann.hpp>

namespace flann
{
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
		
		void knnSearchGpu(const cuda::DeviceMatrix<ElementType>& queries, cuda::DeviceMatrix<int>& indices, cuda::DeviceMatrix<DistanceType>& dists, size_t knn, const SearchParams& params, cudaStream_t stream) const
		{

		}

		int radiusSearchGpu(const cuda::DeviceMatrix<ElementType>& queries, cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists, float radius, const SearchParams& params, cudaStream_t stream = NULL) const
		{

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
	class DynGpuIndex : public GpuIndex<Distance>
	{
	public:
		DynGpuIndex(const IndexParams& params, Distance distance = Distance())
		{
			Index<Distnace>::index_params_ = params;
			//nnIndex = new KDTreeCudaIndex<Distance>()


		}
		DynGpuIndex(const cuda::DeviceMatrix<ElementType> features, const IndexParams& params, Distance distance = Distance())
		{
			Index<Distnace>::index_params_ = params;
			nnIndex_ = new KDTreeCudaIndex<Distance>(features, params, distance);
			nnIndex_->buildIndex();
		}
		virtual int knnSearch(const cuda::DeviceMatrix<ElementType>& queries,
			cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists,
			size_t knn,
			const SearchParams& params,
			cudaStream_t stream = NULL) const
		{
			nnIndex_->knnSearchGpu(queries, indices, dists, knn, params, stream);
			return 0;
		}
		virtual int radiusSearch(const cuda::DeviceMatrix<ElementType>& queries,
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

	};
}