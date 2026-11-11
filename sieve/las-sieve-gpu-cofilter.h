#ifndef CADO_LAS_SIEVE_GPU_COFILTER_H
#define CADO_LAS_SIEVE_GPU_COFILTER_H

/* device side of the cofactor filter (las-sieve-gpu-cofilter.cu); P's td
 * pointers are device pointers, keys the hits of all sides sorted as
 * (x << 33) | (side << 32) | p */

#include <cstdint>
#include <cuda_runtime.h>
#include "las-sieve-gpu-kernels.h"

void lsg_cofilter_launch(cudaStream_t stream, lsg_cofilter const & P,
        lsg_survivor * surv, uint32_t nsurv, const uint64_t * keys,
        uint32_t nkeys, int logI);

#endif
