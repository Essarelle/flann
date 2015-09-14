#pragma once
#include "flann/util/params.h"
#include "flann/util/DeviceMatrix.h"
#include <flann/algorithms/dist.h>
#include <memory>

namespace flann
{
	template<typename T> class KDTreeCudaIndex;

	template<typename Distance>
	class DynGpuIndex
	{
	public:
		typedef typename Distance::ElementType ElementType;
		typedef typename Distance::ResultType DistanceType;
		typedef KDTreeCudaIndex<Distance> IndexType;
		DynGpuIndex(const IndexParams& params, Distance distance = Distance());
		DynGpuIndex(cuda::DeviceMatrix<ElementType> features, const IndexParams& params, Distance distance = Distance());
		~DynGpuIndex();
		virtual void buildIndex();
		virtual int knnSearch(cuda::DeviceMatrix<ElementType> queries,
			cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists,
			size_t knn,
			const SearchParams& params,
			cudaStream_t stream = NULL) const;

		virtual int radiusSearch(cuda::DeviceMatrix<ElementType> queries,
			cuda::DeviceMatrix<int>& indices,
			cuda::DeviceMatrix<DistanceType>& dists,
			float radius,
			const SearchParams& params,
			cudaStream_t stream = NULL) const;
	protected:
		std::shared_ptr<KDTreeCudaIndex<Distance>> nnIndex_;
		IndexParams index_params_;

	}; // class DynGpuIndex
} // namespace flann
