#pragma once
#include <flann/algorithms/DynKdtree_cuda_builder.h>
#include <flann/util/DeviceMatrix.h>
#include <flann/algorithms/kdtree_cuda_3d_index.h> // For params class

namespace flann
{
	template<typename T>
	struct GpuHelper
	{
		thrust::device_vector< cuda::kd_tree_builder_detail::SplitInfo >* gpu_splits_;
		thrust::device_vector< int >* gpu_parent_;
		thrust::device_vector< int >* gpu_child1_;
		flann::cuda::DeviceMatrix<T> gpu_points_;
		flann::cuda::DeviceMatrix<T> gpu_aabb_min_;
		flann::cuda::DeviceMatrix<T> gpu_aabb_max_;
		thrust::device_vector<int>* gpu_vind_;

		GpuHelper() : gpu_splits_(0), gpu_parent_(0), gpu_child1_(0), gpu_vind_(0){}
		~GpuHelper()
		{
			delete gpu_splits_;
			gpu_splits_ = 0;
			delete gpu_parent_;
			gpu_parent_ = 0;
			delete gpu_child1_;
			gpu_child1_ = 0;
			delete gpu_aabb_max_;
			gpu_aabb_max_ = 0;
			delete gpu_aabb_min_;
			gpu_aabb_min_ = 0;
			delete gpu_vind_;
			gpu_vind_ = 0;

			delete gpu_points_;
			gpu_points_ = 0;
		}
	};

	template<typename T, typename Distance>
	class KDTreeCudaIndex: public NNIndex<Distance>
	{
		typedef typename Distance::ElementType ElementType;
		typedef typename Distance::ResultType DistanceType;
		typedef NNIndex<Distance> BaseClass;

		int visited_leafs;
		
		typedef bool needs_kdtree_distance;

		KDTreeCudaIndex(
			const DeviceMatrix<ElementType>& inputData,
			const IndexParams& params = KDTreeCuda3dIndexParams(),
			Distance d = Distance()
			) :
				BaseClass(Params, d),
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
			delete[] data_.ptr();
			uploadTreeToGpu();
		}

	private:
		virtual void uploadTreeToGpu()
		{
			delete gpu_helper_;
			gpu_helper = new GpuHelper<ElementType>;
			::flann::cuda::dyn_kd_tree_builder_detail::CudaKdTreeBuilder<ElementType> builder(dataset_, leaf_max_size_);
			builder.buildTree();
			
			gpu_helper_->gpu_splits_ = builder.splits_;
			gpu_helper_->gpu_aabb_max_ = builder.aabb_max_;
			gpu_helper_->gpu_aabb_min_ = builder.aabb_min_;
			gpu_helper_->gpu_child1_ = builder.child1_;
			gpu_helper_->gpu_parent_ = builder.parent_;
			//thrust::copy(builder.index_.begin(0), builder.index_.end(0), 
			if (gpu_helper_->gpu_vind_ == nullptr)
				gpu_helper_->gpu_vind_ = new thrust::device_vector<int>();

			gpu_helper_->gpu_vind_->insert(gpu_helper_->gpu_vind_->begin(), builder.index_.begin(0), builder.index_.end(0));
			thrust::gather(builder.index_.begin(0), builder.index_.end(0), )
			//gpu_helper_->gpu_vind_ = builder.index_;
		}
		

		GpuHelper<ElementType>* gpu_helper_;

		const DeviceMatrix<ElementType> dataset_;

		int leaf_max_size_;

		int leaf_count_;
		int node_count_;
		//! used by convertTreeToGpuFormat
		int current_node_count_;
		std::vector<int> vind_;

		DeviceMatrix<ElementType> data_;

		size_t dim_;

		USING_BASECLASS_SYMBOLS

	};// KDTreeCudaIndex
}