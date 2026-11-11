#ifndef CADO_SQRT_GPU_MUL_H
#define CADO_SQRT_GPU_MUL_H

/* Multiplication of large integers on the GPU (sqrt_gpu_mul.cu), for the
 * square root's product trees and lifting, whose time is GMP's FFT
 * multiplication of numbers of millions of bits. These are far beyond the
 * fixed widths of CGBN; this is an NTT: the operands are cut into 64-bit
 * pieces, convolved modulo three primes below 2^62 (each convolution
 * coefficient is below n 2^128 < 2^183, the product of the primes),
 * recombined by CRT, and the carries are propagated on the host. The
 * product is exact: the same integer as mpz_mul's.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 0 if there is no usable GPU */
int sgm_init(void);

/* r[0 .. na+nb) = a[0 .. na) * b[0 .. nb), 64-bit limbs, least
 * significant first (na, nb >= 1). Thread-safe (one product at a time on
 * the GPU). */
void sgm_mul(uint64_t * r, const uint64_t * a, size_t na,
        const uint64_t * b, size_t nb);

void sgm_stats(double * seconds, unsigned long * calls);
/* the number of GPUs used (logical ones with CADO_GPU_SIMULATE_NDEV) */
int sgm_ngpu(void);

#ifdef __cplusplus
}
#endif

#endif
