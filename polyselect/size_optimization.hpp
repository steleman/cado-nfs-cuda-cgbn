#ifndef CADO_POLYSELECT_SIZE_OPTIMIZATION_HPP
#define CADO_POLYSELECT_SIZE_OPTIMIZATION_HPP

#include <gmp.h>

#include "mpz_poly.h"

/* Default maximal number of steps in sopt_local_descent */
#define SOPT_DEFAULT_MAX_STEPS 300

/* Default value for sopt_effort */
#define SOPT_DEFAULT_EFFORT 0

/* values of q greater than 1e10 in absolute value do not help */
#define SOPT_MAX_VALUE_FOR_Q_ROOTS 1e10

/* Number of best rational approximations to keep for each q-root */
#define SOPT_NB_RAT_APPROX_OF_Q_ROOTS 16

/* Maximum value for the denominator of rational approximations of q-roots */
#define SOPT_MAX_DEN_IN_RAT_APPROX_OF_Q_ROOTS 100.0

/* Call LLL for skew in [ skew0^(e/(2*NSKEW)) for e in [NSKEW..3*NSKEW] ]*/
#define SOPT_NSKEW 2 /* 2 seems to be quite good */
#define SOPT_NB_OF_SKEWNESS_VALUES (2*SOPT_NSKEW + 1)

/* Only the SOPT_MAX_LLL_POLY_PROCESS best polynomials produced by LLL are
 * given to the local optimization algorithm (should be >= 1) */
#define SOPT_MAX_LLL_POLY_PROCESS 2

/* Maximum degree possible of the rotations in the local descent algorithm. */
#define SOPT_MAX_DEGREE_ROTATION 7

#define SOPT_LOCAL_DESCENT_GUARD 0.001

double sopt_local_descent (cxx_mpz_poly &, cxx_mpz_poly &, cxx_mpz_poly const &,
                           cxx_mpz_poly const &, int, int, unsigned int, int);
double size_optimization (cxx_mpz_poly &, cxx_mpz_poly &, cxx_mpz_poly const &,
                          cxx_mpz_poly const &, unsigned int, int);

/* sopt_local_descent from the state at the beginning of an iteration:
   steps kt = 2^kt_exp, k[i] = 2^k_exp[i], iter iterations done */
double sopt_local_descent_continue (cxx_mpz_poly & f_opt, cxx_mpz_poly & g_opt,
                                    int kt_exp, int const * k_exp,
                                    unsigned int iter, int use_translation,
                                    int deg_rotation, unsigned int max_iter,
                                    int verbose);

/* size_optimization of several pairs at once: the same results as
   size_optimization on each, in order, with the local descents of all of
   them run together (on the GPU when sopt_gpu_descents uses it). */
#include <vector>
struct sopt_batch_item {
    cxx_mpz_poly f_raw, g_raw;  /* input */
    cxx_mpz_poly f_opt, g_opt;  /* output */
    double lognorm = 0;         /* size_optimization's return value */
};
void size_optimization_batch (std::vector<sopt_batch_item> & items,
                              unsigned int sopt_effort, int verbose);

#endif	/* CADO_POLYSELECT_SIZE_OPTIMIZATION_HPP */


