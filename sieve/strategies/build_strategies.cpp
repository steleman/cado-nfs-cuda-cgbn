#include "cado.h" // IWYU pragma: keep

#include <cfloat>
#include <cmath>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "fmt/base.h"

#include "convex_hull.hpp"
#include "factorization_pattern.hpp"
#include "facul_ecm.h"
#include "facul_method.hpp"
#include "cofactoring_method.hpp"
#include "build_strategies.hpp"
#include "macros.h"
#include "cofactoring_strategy.hpp"
#include "utils_cxx.hpp"

/* Chaining PM1 after PP1 (or the other way round) does not give
 * independent events, so the failure probabilities do not simply
 * multiply. This is the correction that has always been applied. */
static bool is_pm1_or_pp1(cofactoring_method const & fm)
{
    return fm.params.method == PM1_METHOD ||
           fm.params.method == PP1_27_METHOD ||
           fm.params.method == PP1_65_METHOD;
}

/************************************************************************/
/*                      COLLECT DATA FOR ONLY ONE COFACTOR              */
/************************************************************************/

/* The probability that 'fm' fails to find a non-trivial factor of a
 * number with this decomposition. */
double compute_proba_method_one_pattern(factorization_pattern const & D,
                                       cofactoring_method const & fm)
{
    double proba_fail = 1;
    for (auto j: D) {
        j -= fm.len_p_min;
        if (j < fm.proba.size())
            proba_fail *= 1 - fm.proba[j];
        // else the probability is close enough to 0.
    }
    return proba_fail;
}

/* The probability that 'strat' finds a non-trivial factor, averaged over
 * the decompositions that fall within [len_p_min, len_p_max]. */
double compute_proba_strategy(factorization_pattern_list const & init_tab,
                              cofactoring_strategy const & strat,
                              unsigned int len_p_min, unsigned int len_p_max)
{
    double all = 0.0;
    double nb_found_elem = 0;

    for (auto const & D: init_tab) {
        if (all_sizes_within(D, len_p_min, len_p_max)) {
            double p_fail_all = 1;
            for (auto const & st: strat.steps) {
                auto const & fm = st.method;
                double const p_fail_one =
                    compute_proba_method_one_pattern(D, fm);
                p_fail_all *= p_fail_one;
                if (is_pm1_or_pp1(fm))
                    p_fail_all = (p_fail_one + p_fail_all) / 2;
            }
            nb_found_elem += (1 - p_fail_all) * D.count;
        }
        all += D.count;
    }
    if (all < (double)LDBL_EPSILON) // no decomposition at all
        return 0;
    return nb_found_elem / all;
}

/* The average time 'strat' takes on a cofactor of r bits. */
double compute_time_strategy(factorization_pattern_list const & init_tab,
                             cofactoring_strategy const & strat, unsigned int r)
{
    double time_average = 0;
    double all = 0.0;

    for (auto const & D: init_tab) {
        double time_dec = 0;
        double proba_fail_all = 1;
        for (auto const & st: strat.steps) {
            auto const & fm = st.method;
            time_dec += fm.time_for(r) * proba_fail_all;

            double const proba_fail_method =
                compute_proba_method_one_pattern(D, fm);
            proba_fail_all *= proba_fail_method;
            if (is_pm1_or_pp1(fm))
                proba_fail_all = (proba_fail_all + proba_fail_method) / 2;
        }

        time_average += time_dec * D.count;
        all += D.count;
    }
    if (all < (double)LDBL_EPSILON) // no decomposition at all
        return 0;

    return time_average / all;
}

/* Append a strategy to 't', dropping the zero methods it holds. A
 * strategy made only of zero methods keeps its first one, so that it is
 * not empty. */
static void add_strategy_without_zero(cofactoring_strategy_list & t,
                                      cofactoring_strategy const & src)
{
    cofactoring_strategy elem;
    bool strat_is_zero = true;
    for (auto const & st: src.steps) {
        if (!st.method.is_zero()) {
            strat_is_zero = false;
            elem.add(st.method, st.side);
        }
    }
    if (strat_is_zero)
        elem.add(src.steps[0].method, src.steps[0].side);

    elem.proba = src.proba;
    elem.time = src.time;
    t.push_back(std::move(elem));
}

/************************************************************************/
/*                   GENERATE MATRIX                                    */
/************************************************************************/

/* Add the different chains of ECM curves to 'strat'. 'lbucket' lets
 * curves be added in buckets of that length. */
static void generate_collect_iter_ecm(
    cofactoring_method_list const & ecm, unsigned int ind_ecm, cofactoring_strategy & strat,
    unsigned int ind_tab, unsigned int index_iter, unsigned int len_iteration,
    unsigned int lbucket, factorization_pattern_list const & init_tab,
    cofactoring_strategy_list & all_strat, unsigned int fbb, unsigned int lpb,
    unsigned int r, int is_already_used_B12M16)
{
    if (index_iter >= len_iteration) {
        add_strategy_without_zero(all_strat, strat);
        cofactoring_strategy & added = all_strat.back();
        added.proba = compute_proba_strategy(init_tab, added, fbb, lpb);
        added.time = compute_time_strategy(init_tab, added, r);
        return;
    }

    for (unsigned int i = ind_ecm; i < ecm.size(); i++) {
        /* BRENT12 and MONTY16 have only one useful sigma, so they are
         * used at most once in a chain. */
        if (ecm[i].params.parameterization == MONTY16 ||
            ecm[i].params.parameterization == BRENT12) {
            if (is_already_used_B12M16)
                continue;
            strat.steps[ind_tab].method = ecm[i];
            generate_collect_iter_ecm(ecm, i + 1, strat, ind_tab + 1,
                                      index_iter + 1, len_iteration, lbucket,
                                      init_tab, all_strat, fbb, lpb, r, true);
        } else { // MONTY12
            for (unsigned int j = 0;
                 j < lbucket && ind_tab + j < strat.steps.size(); j++)
                strat.steps[ind_tab + j].method = ecm[i];
            generate_collect_iter_ecm(ecm, i, strat, ind_tab + lbucket,
                                      index_iter + lbucket, len_iteration,
                                      lbucket + 1, init_tab, all_strat, fbb,
                                      lpb, r, is_already_used_B12M16);
        }
    }

    /* Reduce the collection to its convex hull, so that it does not fill
     * the RAM.
     *
     * The test reads the wrong way round: the sibling code in
     * generate_strategy_r0_r1() prunes when the collection has grown past
     * 100000, whereas this prunes when it has not, which means at nearly
     * every node of the recursion. Flipping it leaves the output of the
     * whole chain unchanged on the pinned test inputs, and a point that
     * is above the hull of a set cannot be on the hull of a larger one,
     * so pruning early is sound. What is not exactly idempotent is the
     * coarsening convex_hull() applies to points that share an x to
     * within a hair: a subset can keep a different representative from
     * the set it came from. Measuring the cost of the present form needs
     * a realistic instance. */
    if (all_strat.size() < 100000)
        all_strat = convex_hull_strategy(all_strat);
}

/* The best strategies for one cofactor size 'r' on one side. The
 * generator chains methods as
 *   PM1 (0/1) + PP1 (0/1) + ECM-M12(0/1/2...) + ECM-M16/B12(0/1) + ECM-M12(...)
 * and keeps only the convex hull as it goes, so that the collection does
 * not fill the RAM.
 */
cofactoring_strategy_list generate_strategies_oneside(factorization_pattern_list const & init_tab,
                                             cofactoring_method const & zero,
                                             cofactoring_method_list const & pm1,
                                             cofactoring_method_list const & pp1,
                                             cofactoring_method_list const & ecm,
                                             int ncurves, unsigned long lim,
                                             unsigned int lpb, unsigned int r)
{
    unsigned int const fbb = ceil(log2((double)(lim + 1)));
    unsigned int const lim_is_prime = 2 * fbb - 1;

    /* fbb is rounded up, so a few sizes from lim_is_prime on may still
     * have no decomposition: no two primes above lim fit in r bits. */
    ASSERT_ALWAYS(init_tab.empty() || r >= lim_is_prime);

    if (init_tab.empty()) {
        /* r is already prime. Two zero strategies cover the cases: a
         * good prime (fbb < r < lpb) succeeds with probability 1, a
         * prime that is too large, or a length that cannot occur, with
         * probability 0. */
        cofactoring_strategy st_zero;
        st_zero.add(zero);
        st_zero.time = 0.0;
        st_zero.proba = (r != 1 && (r < fbb || r > lpb)) ? 0 : 1.0;

        cofactoring_strategy_list res;
        res.push_back(std::move(st_zero));
        return res;
    }

    cofactoring_strategy_list all_strat;

    int const len_strat = 2 + ncurves;
    cofactoring_strategy strat;
    for (int i = 0; i < len_strat; i++)
        strat.add(zero);

    for (auto const & fm_pm1: pm1) {
        double const current_proba_pm1 = fm_pm1.proba[0];
        strat.steps[0].method = fm_pm1;
        for (auto const & fm_pp1: pp1) {
            if (fm_pp1.params.B1 != 0 && fm_pp1.proba[0] < current_proba_pm1)
                continue;
            strat.steps[1].method = fm_pp1;

            generate_collect_iter_ecm(ecm, 0, strat, 2, 0, ncurves, 0, init_tab,
                                      all_strat, fbb, lpb, r, false);
        }
    }
    return convex_hull_strategy(all_strat);
}

/* Concatenate two one-sided strategies, st1 going first on
 * 'first_side'. */
static cofactoring_strategy concat_strategies(cofactoring_strategy const & st1,
                                    cofactoring_strategy const & st2, int first_side)
{
    cofactoring_strategy st;
    for (auto const & step: st1.steps)
        st.add(step.method, first_side);
    for (auto const & step: st2.steps)
        st.add(step.method, first_side ? 0 : 1);
    return st;
}

/* The best strategies for a pair of cofactors, given the optimal
 * strategies for each side. The probability and time of each side must
 * already have been computed. */
cofactoring_strategy_list generate_strategy_r0_r1(cofactoring_strategy_list const & strat_r0,
                                         cofactoring_strategy_list const & strat_r1)
{
    cofactoring_strategy_list strat_r0_r1;
    cofactoring_strategy_list ch;
    unsigned long nb_strat = 0;

    for (size_t r = 0; r < strat_r0.size(); r++) {
        double const p0 = strat_r0[r].proba;
        double const c0 = strat_r0[r].time;
        for (auto const & s1: strat_r1) {
            nb_strat++;
            double const p1 = s1.proba;
            double const c1 = s1.time;
            double const proba = p0 * p1;
            double const tps0 = c0 + p0 * c1; // starting with side 0
            double const tps1 = c1 + p1 * c0; // starting with side 1
            cofactoring_strategy st;
            if (tps0 < tps1) {
                st = concat_strategies(strat_r0[r], s1, 0);
                st.time = tps0;
            } else {
                st = concat_strategies(s1, strat_r0[r], 1);
                st.time = tps1;
            }
            st.proba = proba;
            strat_r0_r1.push_back(std::move(st));
        }

        /* Keep the RAM in check. */
        if (nb_strat > 100000 || r == strat_r0.size() - 1) {
            strat_r0_r1.insert(strat_r0_r1.end(), ch.begin(), ch.end());
            ch = convex_hull_strategy(strat_r0_r1);
            strat_r0_r1.clear();
            nb_strat = 0;
        }
    }

    return ch;
}

/* The best strategies for each pair of cofactor sizes, from a set of
 * factoring methods whose probabilities and times have already been
 * measured (with survey_cofactoring_methods). */
strategy_matrix generate_matrix(std::string const & pattern_dir,
                                cofactoring_method_list const & pm1, cofactoring_method_list const & pp1,
                                cofactoring_method_list const & ecm, int ncurves,
                                unsigned long lim0, unsigned int lpb0,
                                unsigned int mfb0, unsigned long lim1,
                                unsigned int lpb1, unsigned int mfb1)
{
    unsigned int const fbb0 = ceil(log2((double)(lim0 + 1)));
    unsigned int const fbb1 = ceil(log2((double)(lim1 + 1)));

    strategy_matrix matrix(mfb0 + 1);
    for (auto & row: matrix)
        row.resize(mfb1 + 1);

    cofactoring_method const zero = cofactoring_method::from_fields(0, 0, 0, 0);

    auto read_patterns = [&](unsigned long lim, unsigned int r,
                           unsigned int lim_is_prime) {
        factorization_pattern_list patterns;
        if (r >= lim_is_prime) {
            auto filename =
                fmt::format("{}/decomp_{}_{}", pattern_dir, lim, r);
            std::ifstream is(filename);
            if (!(is >> patterns))
                throw cado::error("Cannot read {}", filename);
        }
        return patterns;
    };

    /* Side 0 is precomputed for every size; side 1 is computed as we go. */
    std::vector<cofactoring_strategy_list> data_rat(mfb0 + 1);
    for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
        auto const patterns = read_patterns(lim0, r0, 2 * fbb0 - 1);
        data_rat[r0] = generate_strategies_oneside(patterns, zero, pm1, pp1,
                                                   ecm, ncurves, lim0, lpb0, r0);
    }

    for (unsigned int r1 = 0; r1 <= mfb1; r1++) {
        auto const patterns = read_patterns(lim1, r1, 2 * fbb1 - 1);
        cofactoring_strategy_list const strat_r1 = generate_strategies_oneside(
            patterns, zero, pm1, pp1, ecm, ncurves, lim1, lpb1, r1);

        for (unsigned int r0 = 0; r0 <= mfb0; r0++)
            matrix[r0][r1] = generate_strategy_r0_r1(data_rat[r0], strat_r1);
    }

    return matrix;
}

/************************************************************************/
/*                      CONVEX_HULL_ST                                  */
/************************************************************************/

/* The strategies on the trade-off frontier: those for which no other
 * strategy is both more likely to find a relation and faster. */
cofactoring_strategy_list convex_hull_strategy(cofactoring_strategy_list const & t)
{
    return pareto_reduce(t, [](cofactoring_strategy const & s) {
        return std::make_pair(s.proba, s.time);
    });
}
