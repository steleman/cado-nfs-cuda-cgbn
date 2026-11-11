#ifndef CADO_SOPT_GPU_KERNELS_H
#define CADO_SOPT_GPU_KERNELS_H

/* GPU size optimization (size_optimization.cpp). C interface.
 *
 * The local descent of sopt_local_descent compares values of
 * L2_skew_lognorm. Everything in L2_skew_lognorm is IEEE arithmetic
 * (+ - * /, sqrt, fma, ldexp) that the GPU reproduces exactly (built with
 * -fmad=false, in the same order), except the final log. The GPU uses
 * its own log, and takes a decision only when the two values it compares
 * are further apart than the possible difference between the two logs;
 * otherwise it gives the descent back to the CPU. So every descent the
 * GPU finishes follows the CPU's path exactly.
 *
 * Integers are fixed-width two's complement (nl 64-bit limbs, least
 * significant first); a value that would not fit also gives the descent
 * back to the CPU.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ssg_ctx;

struct ssg_ctx * ssg_ctx_create(void);  /* NULL if there is no GPU */
void ssg_ctx_free(struct ssg_ctx *);

/* status of a result */
enum {
    SSG_OK = 0,
    SSG_CLOSE = 1,      /* two compared values too close to decide */
    SSG_OVERFLOW = 2,   /* an integer does not fit */
    SSG_NONFINITE = 3,  /* inf or nan (the CPU throws there) */
    SSG_NOROOT = 4,     /* the skewness equation had no positive root */
    SSG_RESUME = 5,     /* a close comparison in the main loop: the state
                           at the beginning of that iteration is returned,
                           for sopt_local_descent_continue */
};

/* the state returned with SSG_RESUME: iter, kt_exp, k_exp[0..7] */
#define SSG_STATE_INTS 10

/* Test of the floating-point part: for n polynomials of degree deg
 * (coefficients as above, n * (deg+1) * nl limbs), the skewness that
 * L2_skewness(mpz_poly) returns, and the argument tt of the log in
 * L2_lognorm at that skewness together with fix, so that
 * L2_skew_lognorm(f) = fma(fix, log(2)/2, log(tt)/2). */
void ssg_l2_test(struct ssg_ctx *, int n, int deg, int nl,
        const uint64_t * coeffs, double * skew, double * tt, int * fix,
        double * approx, int * status);

/* Local descents (sopt_local_descent with use_translation = 1): n pairs
 * (f of degree deg, g of degree 1), deg_rotation, max_iter. On SSG_OK,
 * f and g are replaced by the result (f_opt, g_opt); on SSG_RESUME, by
 * the state at the beginning of the iteration where the GPU stopped,
 * with state[SSG_STATE_INTS * i ...] = iter, kt_exp, k_exp[]; otherwise
 * they are left alone and the CPU must do that descent. */
void ssg_descent(struct ssg_ctx *, int n, int deg, int nl, uint64_t * f,
        uint64_t * g, int deg_rotation, unsigned int max_iter, int * status,
        int * state);

/* the device's FP32 / FP64 throughput ratio (2 on H100/B200, 64 on
 * consumer and workstation parts) */
int ssg_fp64_ratio(const struct ssg_ctx *);

void ssg_ctx_stats(const struct ssg_ctx *, double * seconds,
        unsigned long * descents, unsigned long * returned);

#ifdef __cplusplus
}
#endif

#endif
