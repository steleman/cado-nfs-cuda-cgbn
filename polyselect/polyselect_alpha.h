#ifndef CADO_POLYSELECT_ALPHA_H
#define CADO_POLYSELECT_ALPHA_H

#include <gmp.h>

#include "mpz_poly.h"
#ifdef __cplusplus
extern "C" {
#endif

/* default prime bounds for the computation of alpha */
#define ALPHA_BOUND_SMALL  100
#ifndef ALPHA_BOUND /* allows to define ALPHA_BOUND in local.sh */
#define ALPHA_BOUND       2000
#endif

extern double get_alpha (mpz_poly_srcptr f, unsigned long B);

/* An optional, per-thread source of root counts for get_alpha: given f
 * and B, it returns NULL, or an array whose k-th entry is the number of
 * distinct roots of f modulo the k-th prime <= B (2, 3, 5, ...). get_alpha
 * then uses these counts where it would count the roots itself (primes
 * whose square does not divide disc(f)); nothing else changes. Used by
 * polyselect_ropt to count roots on the GPU. NULL removes the source. */
typedef const unsigned short * (*alpha_rootcount_fn) (mpz_poly_srcptr f,
        unsigned long B, void * arg);
extern void set_alpha_rootcount_source (alpha_rootcount_fn fn, void * arg);

extern double get_alpha_projective (mpz_poly_srcptr f, unsigned long B);

extern double get_alpha_affine_p(mpz_poly_srcptr f, unsigned long p,
                          gmp_randstate_ptr rstate);

extern void set_alpha_bound (unsigned long bound);

extern unsigned long get_alpha_bound (void);

extern double expected_alpha (double logK);

#ifdef __cplusplus
}
#endif

#endif	/* CADO_POLYSELECT_ALPHA_H */
