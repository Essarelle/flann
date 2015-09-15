#include "kdtree_cuda_index_private.cuh"
template class DynGpuIndex<L1<double>>;
template class DynGpuIndex<L1<float>>;
template class DynGpuIndex<L1<int>>;
template class DynGpuIndex<L1<char>>;
template class DynGpuIndex<L1<short>>;
template class DynGpuIndex<L1<unsigned int>>;
template class DynGpuIndex<L1<unsigned short>>;
template class DynGpuIndex<L1<unsigned char>>;
