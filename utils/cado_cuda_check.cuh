#ifndef CADO_CUDA_CHECK_CUH
#define CADO_CUDA_CHECK_CUH

/* Whether this binary has GPU code that the current device can run.
 * A binary built for other GPUs (CMAKE_CUDA_ARCHITECTURES=native on a
 * development card, then copied to a B200) would otherwise fail at its
 * first kernel launch. The probe kernel is compiled into each CUDA file
 * with the same architectures as its other kernels. */

#include <cstdio>
#include <cuda_runtime.h>

namespace {

__global__ void cado_cuda_probe_kernel() {}

/* who: the module, for the message (printed once per module) */
inline bool cado_cuda_can_run(const char * who)
{
    cudaFuncAttributes a;
    cudaError_t const e = cudaFuncGetAttributes(&a, cado_cuda_probe_kernel);
    if (e == cudaSuccess) return true;
    (void) cudaGetLastError();
    static bool said = false;
    if (!said) {
        said = true;
        int dev = -1, maj = 0, mnr = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
        fprintf(stderr, "# %s: this binary has no GPU code for device %d"
                " (compute capability %d.%d: %s); using the CPU. Rebuild with"
                " CMAKE_CUDA_ARCHITECTURES (or CADO_CUDA_ARCHITECTURES)"
                " including %d%d.\n", who, dev, maj, mnr,
                cudaGetErrorString(e), maj, mnr);
    }
    return false;
}

}  /* namespace */

#endif
