#ifndef CADO_ROPT_GPU_KERNELS_H
#define CADO_ROPT_GPU_KERNELS_H

/* GPU root sieve for polyselect_ropt (ropt_stage2.cpp). C interface.
 *
 * The CPU's sieve subtracts sub from array[j] for j = start, start +
 * stride, ... <= bound, for a list of such lines; the GPU applies the
 * same lines to an array that starts at init everywhere. The result is
 * the same: the subtractions commute (modulo 2^16, like the CPU's
 * int16_t arithmetic). */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct rsg_op {
    int64_t start, bound;   /* positions in the array */
    uint32_t stride;
    int32_t sub;
};

struct rsg_ctx;

struct rsg_ctx * rsg_ctx_create(void);  /* NULL if there is no GPU */
void rsg_ctx_free(struct rsg_ctx *);

/* the op buffer, pinned host memory with room for n ops */
struct rsg_op * rsg_ops_buffer(struct rsg_ctx *, size_t n);

/* applies the n ops in the buffer; out[0 .. len) gets the array */
void rsg_apply(struct rsg_ctx *, size_t n, int16_t init, int16_t * out,
        size_t len);

void rsg_ctx_stats(const struct rsg_ctx *, double * seconds,
        unsigned long * calls, unsigned long * ops);

/* Root counts for get_alpha: for each of npolys polynomials of degree
 * <= deg, whose coefficient i is (-1)^sign[..] times the nlimbs 32-bit
 * limbs (least significant first) at limbs[((poly * (deg+1)) + i) *
 * nlimbs], and each of the nprimes primes (< 2^16), the number of x in
 * [0, p) with f(x) = 0 mod p (the number of distinct roots of f mod p
 * when f mod p is not 0), in counts[poly * nprimes + k]. */
void rsg_rootcounts(struct rsg_ctx *, int npolys, int deg, int nlimbs,
        const uint32_t * limbs, const uint8_t * sign,
        int nprimes, const uint32_t * primes, unsigned short * counts);

#ifdef __cplusplus
}
#endif

#endif
