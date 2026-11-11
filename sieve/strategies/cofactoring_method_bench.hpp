#ifndef CADO_COFACTORING_METHOD_BENCH_HPP
#define CADO_COFACTORING_METHOD_BENCH_HPP

#include <array>
#include <istream>
#include <ostream>
#include <vector>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "facul.hpp"
#include "facul_ecm.h"
#include "cofactoring_method.hpp"
#include "cofactoring_strategy.hpp"
#include "convex_hull.hpp"

/************************************************************************/
/*                      To generate numbers n=pq*/
/************************************************************************/

std::vector<double> distribution_prime_number(int min, int max);

cxx_mpz generate_prime_factor(gmp_randstate_t state, int lenFact);

cxx_mpz
generate_composite_integer(gmp_randstate_t state,
			   int lenFact1, int lenFactall);

cxx_mpz
generate_composite_integer_interval(gmp_randstate_t state,
				    std::vector<double> const & dist,
				    int lenFact1_min, int lenFactall);

/************************************************************************/
/*                To model our factoring methods                        */
/************************************************************************/

/* One-sided facul strategies holding a single method each, one per curve
 * of the family that a measurement is to be spread over. For a non-ECM
 * method that is one strategy; for ECM it is several generic curves,
 * skipping the exceptional parameters, since a method named without a
 * parameter means a generic one. extra_primes is set to 1. */
std::vector<facul_strategy_oneside> generate_fm (facul_method_code method,
                               unsigned long B1, unsigned long B2,
                               ec_parameterization_t curve);

/************************************************************************/
/*                      Create Factoring Methods */
/************************************************************************/

std::array<int, 6> choice_parameters(facul_method_code method, int len_p_min);

cofactoring_method_list bench_proba_time_pset(facul_method_code method, ec_parameterization_t curve,
				    gmp_randstate_t state,
				    int len_p_min, int len_p_max,
				    int len_n, const int *param_region);

cofactoring_method_list generate_factoring_methods(gmp_randstate_t state, int len_p_min,
					 int len_p_max, int len_n, int opt_ch,
					 const int *param_sieve);

cofactoring_method_list generate_factoring_methods_mc(gmp_randstate_t state,
					    int len_p_min, int len_p_max,
					    int len_n, facul_method_code method, ec_parameterization_t curve,
					    int opt_ch,
                                            const int *param_sieve);

/************************************************************************/
/*                      ANALYSE AND FILTER */
/************************************************************************/

double bench_proba_fm(std::vector<facul_strategy_oneside> const & curves,
        gmp_randstate_t state,
 		      unsigned long len_p, unsigned long len_n,
                      std::vector<cxx_mpz> & N,
		      size_t nb_test_max);

void bench_proba(gmp_randstate_t state, cofactoring_method_list & fm, int len_p_min,
        int p_max, size_t nb_test_max);


double bench_time_fm_onelength(std::vector<facul_strategy_oneside> const & curves,
                               std::vector<cxx_mpz> & N,
			       size_t nb_test);

void bench_time(gmp_randstate_t state, cofactoring_method_list & fm, size_t nb_test);


cofactoring_method_list filtering(cofactoring_method_list const & fm, int final_nb_methods);

/************************************************************************/
/*                      CONVEX_HULL_FM                                  */
/************************************************************************/

cofactoring_method_list convex_hull_fm(cofactoring_method_list const & t);

/************************************************************************/
/*               COMPUTE THE CONVEX_HULL FROM A FILE OF FM              */
/************************************************************************/

cofactoring_method_list convex_hull_from_file(std::istream & in, std::ostream & out);

struct weighted_success {
    double prob = 0;
    double time = 0;
    // weighted_success() = default;
    weighted_success(double p, double t) : prob(p), time(t) {}
    weighted_success(size_t p, double t, size_t N)
        : prob(double(p)/double(N))
        , time(t/double(N))
    {}
};

#endif /* CADO_FACTORING_METHOD_BENCH_HPP */
