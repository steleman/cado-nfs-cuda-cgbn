#ifndef CADO_POLYSELECT_GPU_KERNELS_H
#define CADO_POLYSELECT_GPU_KERNELS_H

/* GPU collision test for polyselect's special-q stage (Kleinjung's
 * algorithm as in polyselect_collisions.cpp). C interface.
 *
 * For a special-q q (a product of small primes) and one of its roots rq
 * (mod q^2), every root r of (m0+x)^d = N (mod p^2), p a prime of the
 * table, gives the values u = (r - rq) / q^2 (mod p^2) in [-M, M). The
 * CPU's decisional pass (polyselect_DCS_flat_subtask) says whether two
 * of these values are equal; this does the same test for many (q, rq)
 * at once, comparing the values exactly. A flag can also be set when a
 * block runs out of room (or q^2 is not invertible mod some p^2): the
 * caller then runs the CPU passes on that rq, which decide exactly. So a
 * clear flag means no collision, the same answer as the CPU's.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct psg_ctx;

/* NULL if there is no usable GPU */
struct psg_ctx * psg_ctx_create(void);
void psg_ctx_free(struct psg_ctx *);

/* the roots of one ad: nprimes primes with roots (p < 2^31), their
 * roots modulo p^2 (nroots, each with the index of its prime), and M.
 * Returns 0 if these cannot be handled (the caller uses the CPU). */
int psg_set_roots(struct psg_ctx *, const uint32_t * primes, uint32_t nprimes,
        const uint64_t * roots, const uint32_t * root_pidx, uint32_t nroots,
        int64_t umax);

/* nq distinct special-q values; nrq roots, rq[2*i] + 2^64 rq[2*i+1],
 * of special-q qs[rq_qidx[i]]. Sets flags[i] to 1 when rq i may have a
 * collision (see above), 0 when it has none. */
void psg_test(struct psg_ctx *, const uint64_t * qs, uint32_t nq,
        const uint64_t * rq, const uint32_t * rq_qidx, uint32_t nrq,
        uint8_t * flags);

void psg_ctx_stats(const struct psg_ctx *, double * seconds,
        unsigned long * rqs, unsigned long * flagged, unsigned long * overflows);

#ifdef __cplusplus
}
#endif

#endif /* CADO_POLYSELECT_GPU_KERNELS_H */
