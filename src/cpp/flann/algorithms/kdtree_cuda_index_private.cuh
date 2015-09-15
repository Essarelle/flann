#include "kdtree_cuda_index.cuh"
#include "kdtree_cuda_index_impl.cuh"
using namespace flann;

template<typename Distance> DynGpuIndex<Distance>::DynGpuIndex(const IndexParams& params, Distance distance = Distance())
{
	index_params_ = params;
}
template<typename Distance> DynGpuIndex<Distance>::DynGpuIndex(cuda::DeviceMatrix<ElementType> features, const IndexParams& params, Distance distance = Distance())
{
	index_params_ = params;
	nnIndex_.reset(new KDTreeCudaIndex<Distance>(features, params, distance));
}
template<typename Distance> DynGpuIndex<Distance>::~DynGpuIndex()
{

}
template<typename Distance> void DynGpuIndex<Distance>::buildIndex()
{
	nnIndex_->buildIndex();
}
template<typename Distance> int DynGpuIndex<Distance>::knnSearch(cuda::DeviceMatrix<ElementType> queries,
	cuda::DeviceMatrix<int>& indices,
	cuda::DeviceMatrix<DistanceType>& dists,
	size_t knn,
	const SearchParams& params,
	cudaStream_t stream = NULL) const
{
	nnIndex_->knnSearchGpu(queries, indices, dists, knn, params, stream);
	return 0;
}
template<typename Distance>  int DynGpuIndex<Distance>::radiusSearch(cuda::DeviceMatrix<ElementType> queries,
	cuda::DeviceMatrix<int>& indices,
	cuda::DeviceMatrix<DistanceType>& dists,
	float radius,
	const SearchParams& params,
	cudaStream_t stream = NULL) const
{
	return nnIndex_->radiusSearchGpu(queries, indices, dists, radius, params, stream);
}