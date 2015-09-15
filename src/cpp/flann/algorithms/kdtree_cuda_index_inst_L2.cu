#include "kdtree_cuda_index_private.cuh"

template class DynGpuIndex<L2<double>>;
template class DynGpuIndex<L2<float>>;
template class DynGpuIndex<L2<int>>;
template class DynGpuIndex<L2<char>>;
template class DynGpuIndex<L2<short>>;
template class DynGpuIndex<L2<unsigned int>>;
template class DynGpuIndex<L2<unsigned short>>;
template class DynGpuIndex<L2<unsigned char>>;