#ifndef CADO_FACTORIZATION_PATTERNS_HPP
#define CADO_FACTORIZATION_PATTERNS_HPP

#include <ostream>

#include "factorization_pattern.hpp"
#include "cofactoring_strategy.hpp"

/* The ways a cofactor of mfb bits, with no prime factor below lim, can
 * factor -- each with an estimate of how many such cofactors factor that
 * way. This is what tells the strategy generator which prime sizes it has
 * to be able to find.
 *
 * The estimate comes from the generating series of the product; see the
 * comments at the top of factorization_patterns.cpp for the two methods
 * that were compared.
 */
factorization_pattern_list list_factorization_patterns(unsigned int mfb,
                                                       unsigned long lim);

/* The same counts, computed both ways -- by the generating series and by
 * sampling -- side by side with their ratio. Useful to check that the
 * cheap estimate is close enough to the expensive one. */
void compare_pattern_estimates(unsigned int mfb, unsigned long lim,
                               unsigned long seed, std::ostream & os);

#endif /* CADO_FACTORIZATION_PATTERNS_HPP */
