#ifndef CADO_BUILD_STRATEGIES_HPP
#define CADO_BUILD_STRATEGIES_HPP

#include <string>
#include <vector>

#include "factorization_pattern.hpp"
#include "cofactoring_method.hpp"
#include "cofactoring_strategy.hpp"
#include "convex_hull.hpp"

/************************************************************************/
/*                      COLLECT DATA FOR ONLY ONE COFACTOR              */
/************************************************************************/

double compute_proba_method_one_pattern(factorization_pattern const & dec,
                                       cofactoring_method const & fm);

double compute_proba_strategy(factorization_pattern_list const & init_tab,
                              cofactoring_strategy const & strat,
                              unsigned int len_p_min, unsigned int len_p_max);

double compute_time_strategy(factorization_pattern_list const & init_tab,
                             cofactoring_strategy const & strat, unsigned int r);

/************************************************************************/
/*                   GENERATE MATRIX                                    */
/************************************************************************/

cofactoring_strategy_list generate_strategies_oneside(factorization_pattern_list const & patterns,
                                             cofactoring_method const & zero,
                                             cofactoring_method_list const & pm1,
                                             cofactoring_method_list const & pp1,
                                             cofactoring_method_list const & ecm,
                                             int nb_curve, unsigned long lim,
                                             unsigned int lpb, unsigned int r);

cofactoring_strategy_list generate_strategy_r0_r1(cofactoring_strategy_list const & strat_r0,
                                         cofactoring_strategy_list const & strat_r1);

/* matrix[r0][r1], with 0 <= r0 <= mfb0 and 0 <= r1 <= mfb1 */
using strategy_matrix = std::vector<std::vector<cofactoring_strategy_list>>;

strategy_matrix generate_matrix(std::string const & pattern_dir,
                                cofactoring_method_list const & pm1, cofactoring_method_list const & pp1,
                                cofactoring_method_list const & ecm, int nb_curve,
                                unsigned long lim0, unsigned int lpb0,
                                unsigned int mfb0, unsigned long lim1,
                                unsigned int lpb1, unsigned int mfb1);

/************************************************************************/
/*                      CONVEX_HULL_ST                                  */
/************************************************************************/

cofactoring_strategy_list convex_hull_strategy(cofactoring_strategy_list const & t);

#endif /* CADO_BUILD_STRATEGIES_HPP */
