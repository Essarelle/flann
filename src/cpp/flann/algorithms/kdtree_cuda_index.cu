#include "kdtree_cuda_index.cuh"
#include "kdtree_cuda_index_impl.cuh"
using namespace flann;

template<typename T> DynGpuIndex<T>::DynGpuIndex(const IndexParams& params, Distance distance = Distance())
{
	index_params_ = params;
}
template<typename T> DynGpuIndex<T>::DynGpuIndex(cuda::DeviceMatrix<ElementType> features, const IndexParams& params, Distance distance = Distance())
{
	index_params_ = params;
	nnIndex_.reset(new KDTreeCudaIndex<Distance>(features, params, distance));
}
template<typename T> DynGpuIndex<T>::~DynGpuIndex()
{

}
template<typename T> void DynGpuIndex<T>::buildIndex()
{
	nnIndex_->buildIndex();
}
template<typename T> int DynGpuIndex<T>::knnSearch(cuda::DeviceMatrix<ElementType> queries,
	cuda::DeviceMatrix<int>& indices,
	cuda::DeviceMatrix<DistanceType>& dists,
	size_t knn,
	const SearchParams& params,
	cudaStream_t stream = NULL) const
{
	nnIndex_->knnSearchGpu(queries, indices, dists, knn, params, stream);
	return 0;
}
template<typename T>  int DynGpuIndex<T>::radiusSearch(cuda::DeviceMatrix<ElementType> queries,
	cuda::DeviceMatrix<int>& indices,
	cuda::DeviceMatrix<DistanceType>& dists,
	float radius,
	const SearchParams& params,
	cudaStream_t stream = NULL) const
{
	return nnIndex_->radiusSearchGpu(queries, indices, dists, radius, params, stream);
}
template<> class DynGpuIndex<L2<double>>;
template<> class DynGpuIndex<L2<float>>;
template<> class DynGpuIndex<L2<int>>;
template<> class DynGpuIndex<L2<char>>;
template<> class DynGpuIndex<L2<short>>;
template<> class DynGpuIndex<L2<unsigned int>>;
template<> class DynGpuIndex<L2<unsigned short>>;
template<> class DynGpuIndex<L2<unsigned char>>;

template<> class DynGpuIndex<L1<double>>;
template<> class DynGpuIndex<L1<float>>;
template<> class DynGpuIndex<L1<int>>;
template<> class DynGpuIndex<L1<char>>;
template<> class DynGpuIndex<L1<short>>;
template<> class DynGpuIndex<L1<unsigned int>>;
template<> class DynGpuIndex<L1<unsigned short>>;
template<> class DynGpuIndex<L1<unsigned char>>;
