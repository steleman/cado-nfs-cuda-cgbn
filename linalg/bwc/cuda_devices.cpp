/* Prints the number of usable CUDA devices (0 if none or on any error).
 * bwc.pl runs it to decide whether the linear algebra goes to the GPU.
 *
 * CADO_BWC_SIMULATE_GPUS=<n> (testing only) reports n devices when there
 * is at least one; the matrices then share the real ones (device choice
 * is modulo the real count) and NCCL is allowed several ranks per GPU.
 * That exercises the multi-GPU code on one card, without any speedup. */
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

int main()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        n = 0;
    const char * sim = getenv("CADO_BWC_SIMULATE_GPUS");
    if (n > 0 && sim && atoi(sim) > 0)
        n = atoi(sim);
    printf("%d\n", n);
    return 0;
}
