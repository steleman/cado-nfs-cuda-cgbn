#ifndef CADO_FACUL_GPU_KERNELS_H
#define CADO_FACUL_GPU_KERNELS_H

/* C interface of the GPU cofactorization kernels (facul-gpu-kernels.cu),
 * so that the C++ side needs no CUDA headers. See facul-gpu.hpp. */

#include <stddef.h>

#include "facul-gpu-data.h"

#ifdef __cplusplus
extern "C" {
#endif

struct fgpu_ctx;

int fgpu_device_count(void);

/* device < 0: CADO_LAS_GPU_DEVICE, else the MPI / SLURM local rank, else
 * 0 (modulo the number of devices). NULL if there is no GPU. */
struct fgpu_ctx * fgpu_create(int device);
void fgpu_free(struct fgpu_ctx *);
int fgpu_device(const struct fgpu_ctx *);

/* the chain description (facul-gpu-data.h); replaces the previous one */
void fgpu_set_blob(struct fgpu_ctx *, const void * blob, size_t bytes);

/* runs n jobs whose cofactors all fit in `words` 64-bit words (1 or 2);
 * synchronous. One caller at a time. */
void fgpu_run(struct fgpu_ctx *, const struct fgpu_job * jobs,
        struct fgpu_result * res, size_t n, int words);

void fgpu_stats(const struct fgpu_ctx *, double * kernel_seconds,
        unsigned long * njobs);

#ifdef __cplusplus
}
#endif

#endif /* CADO_FACUL_GPU_KERNELS_H */
