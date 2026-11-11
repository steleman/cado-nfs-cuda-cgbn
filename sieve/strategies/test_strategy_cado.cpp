#include "cado.h" // IWYU pragma: keep
/*
  This binary allows to test our procedure choosing optimal
  strategies. In fact, using the cofactoring_strategy of CADO, we can use it to
  approximate the "theoretical" number of relations found per second found
  with certains parameters. Then, comparing the "theoretical" value with the
  real value, we could detect if a problem exists in our procedure.
*/

/* There's some dead code in here, which is never called. If we ever come
 * back to this strategy business, there might be something to look at in
 * here, but the odds are low.
 */
#define COMPILE_DEAD_CODE
// #define CADO_INTERLEAVING

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "factorization_pattern.hpp"
#include "facul.hpp"
#include "facul_ecm.h"
#include "facul_method.hpp"
#include "facul_strategies.hpp"
#include "select_best_strategies.hpp"
#include "cofactoring_method.hpp"
#include "cofactoring_method_bench.hpp"
#include "build_strategies.hpp"
#include "gmp_aux.h"
#include "macros.h"
#include "misc.h"
#include "arith/modredc_ul.h"
#include "params.hpp"
#include "cofactoring_strategy.hpp"
#include "timing.h"
#include "cado_main.hpp"
#include "utils_cxx.hpp"
// #include "facul_fwd.hpp"
// #include "pm1.hpp"
// #include "pp1.hpp"
// #include "stage2.h"

#ifdef COMPILE_DEAD_CODE
// #include "facul.hpp"
#endif

// int CONST_TEST_R = 55;


/*
This function generates the cofactoring_strategy of cado and computes the
  probability and the time to find a prime divisor in a cofactor of
  'r' bits with the bound fbb and lpb.  This cofactoring_strategy is the
  concatenation of all methods in 'methods'.
*/
static cofactoring_strategy_list
generate_strategy_cado(cofactoring_method_list const & methods, factorization_pattern_list const & tab_dec,
                       unsigned int fbb, unsigned int lpb, unsigned int r)
{
    cofactoring_strategy strat;

    unsigned int const lim = 2 * fbb - 1;

    ASSERT_ALWAYS((tab_dec.empty()) == (r < lim));

    if (r < lim) {
        strat.add(cofactoring_method::from_fields(PM1_METHOD, 0, 0, 0));
        strat.time = 0.0;
        if (r != 1 && (r < fbb || r > lpb))
            strat.proba = 0.0;
        else // r==1 or fbb<= r0 <= lpb
            strat.proba = 1.0;
    } else {
        /* we set mfb = 3*lpb to avoid the special case of 2 large primes */
        unsigned int const len = 3 + nb_curves(lpb, 3 * lpb);
        ASSERT(len <= methods.size());
        for (unsigned int i = 0; i < len; i++)
            strat.add(methods[i]);

        // eval
        strat.proba = compute_proba_strategy(tab_dec, strat, fbb, lpb);
        strat.time = compute_time_strategy(tab_dec, strat, r);
    }

    cofactoring_strategy_list tab_strat;
    tab_strat.push_back(std::move(strat));
    return tab_strat;
}

/*
  generate the matrix with the cofactoring_strategy of CADO.
*/

static strategy_matrix
generate_matrix_cado(std::string const & pattern_dir,
                     cofactoring_method_list const & methods, unsigned long lim0,
                     unsigned int lpb0, unsigned int mfb0, unsigned long lim1,
                     unsigned int lpb1, unsigned int mfb1)
{
    strategy_matrix matrix(mfb0 + 1);
    for (auto & row: matrix)
        row.resize(mfb1 + 1);

    unsigned int const fbb0 = ceil(log2((double)(lim0 + 1)));
    unsigned int const fbb1 = ceil(log2((double)(lim1 + 1)));

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
    for (unsigned int r0 = 0; r0 <= mfb0; r0++)
        data_rat[r0] = generate_strategy_cado(
            methods, read_patterns(lim0, r0, 2 * fbb0 - 1), fbb0, lpb0, r0);

    for (unsigned int r1 = 0; r1 <= mfb1; r1++) {
        cofactoring_strategy_list const strat_r1 = generate_strategy_cado(
            methods, read_patterns(lim1, r1, 2 * fbb1 - 1), fbb1, lpb1, r1);

        for (unsigned int r0 = 0; r0 <= mfb0; r0++)
            matrix[r0][r1] = generate_strategy_r0_r1(data_rat[r0], strat_r1);
    }

    return matrix;
}

/************************************************************************/
/*                  To interleave our strategies                        */
/************************************************************************/

// problem if tab_edc == nullptr;
static double
compute_time_strategy_ileav(std::array<factorization_pattern_list, 2> const & init_tab,
                            cofactoring_strategy * strat, unsigned int * fbb,
                            unsigned int * lpb, unsigned int * r)
{
    unsigned int const nb_fm = strat->steps.size();
    auto const & tab_fm = strat->steps;

    /* This model costs each method individually, so it needs the
     * per-method timings. A strategy read back from a one-sided strategy
     * file does not have them: that format keeps the chain and the
     * strategy's own (probability, time), and drops the measurements of
     * the methods it is made of. Say so rather than read past the end of
     * an empty vector, which is what used to happen, and rather than
     * charge nothing for every method, which would still print a figure
     * of merit and would be worse. */
    for (auto const & step: strat->steps) {
        if (step.method.params.is_null() || !step.method.time.empty())
            continue;
        throw cado::error("the interleaved cost model needs per-method"
                " timings, and the strategies it was given carry none");
    }

    //{{
    unsigned int last_method_side[2] = {0, 0};
    for (unsigned int index_fm = 0; index_fm < nb_fm; index_fm++) {
        int const side = strat->steps[index_fm].side;
        last_method_side[side] = index_fm;
    }
    unsigned int const end_of_one_side = (last_method_side[0] < last_method_side[1])
                                    ? last_method_side[0]
                                    : last_method_side[1];

    //}}

    //{{
    /*
      We add 0.5 to the length of one word, because for our times the
      length is inclusive. For example, if MODREDCUL_MAXBITS = 64
      bits, a cofactor is in one word if its length is less OR equal to
      64 bits. So, if you don't add 0.5 to MODREDCUL_MAXBITS, you
      lost the equal and thus insert an error in your maths.
    */
    unsigned int ind_time[2];
    for (int side = 0; side < 2; side++) {
        double const half_word = (MODREDCUL_MAXBITS + 0.5) / 2.0;
        int const number_half_wd = floor(r[side] / half_word);
        // the next computation is necessary in the relation with the
        // benchmark in survey_cofactoring_methods!
        ind_time[side] = (number_half_wd < 2) ? 0 : number_half_wd - 1;
    }
    //}}
    // double prob = 0;
    double time_average = 0;
    // store the number of elements in the different decompositions!
    double all = 0.0;
    for (auto const & D0: init_tab[0]) {
        for (auto const & D1: init_tab[1]) {
            double time_dec = 0;
            double proba_fail_side[2] = {
                1, 1}; // didn't find a non-trivial factor!

            double proba_run_next_fm = 1;

            // to know if a side should be consider or not!
            int const is_bad_dec[2] = {!all_sizes_within(D0, fbb[0], lpb[0]),
                                       !all_sizes_within(D1, fbb[1], lpb[1])};

            double time_method = 0;

            // compute the time of each decomposition
            for (unsigned int index_fm = 0; index_fm < nb_fm; index_fm++) {
                cofactoring_method const & elem = tab_fm[index_fm].method;
                int const side = strat->steps[index_fm].side;

                // probability to run the next method!
                /*the next method is used if:
                  - the previous method of the same side did not find a factor!
                  - you didn't have a failure in the other side
                  (i.e.: find a bad decomposition, or find anything
                  and it was our last method for this side!).
                */

                if (is_bad_dec[(1 + side) % 2] == 1)
                    proba_run_next_fm = proba_fail_side[side] *
                                        (proba_fail_side[(1 + side) % 2]);
                // the side will continue if the other side is smooth
                //(proba=1-proba echec)!
                else if (index_fm > last_method_side[(1 + side) % 2] &&
                         is_bad_dec[(1 + side) % 2] == 0)
                    proba_run_next_fm = proba_fail_side[side] *
                                        (1 - proba_fail_side[(1 + side) % 2]);

                else
                    proba_run_next_fm = proba_fail_side[side];

                unsigned int const len_time = elem.time.size();
                if (len_time == 0)
                    time_method = 0;    /* a null method costs nothing */
                else if (ind_time[side] >= len_time)
                    time_method = elem.time[len_time - 1];
                else
                    time_method = elem.time[ind_time[side]];

                time_dec += time_method * proba_run_next_fm;

                double const proba_fail_method =
                    compute_proba_method_one_pattern(side ? D1 : D0, elem);

                proba_fail_side[side] *= proba_fail_method;

                if (elem.params.method == PM1_METHOD ||
                    elem.params.method == PP1_27_METHOD ||
                    elem.params.method == PP1_65_METHOD) {
                    // because if you chain PP1||PM1 to PM1||PP1-->they are
                    // not independant.
                    proba_fail_side[side] =
                        (proba_fail_side[side] + proba_fail_method) / 2;
                }
                if (index_fm == end_of_one_side) {
                    // this side is a bad decomposition in all cases!
                    if (is_bad_dec[side] == 1)
                        proba_fail_side[side] = 0;
                }
            }
            double const count = D1.count * D0.count;
            time_average += time_dec * count;
            /* int is_good[2] = {all_sizes_within (dec[0], fbb[0], lpb[0]), */
            /* 		    all_sizes_within (dec[1], fbb[1], lpb[1])};	   */
            /* prob += nb_elem * (1-proba_fail_side[0]) * is_good[0]* */
            /*   (1-proba_fail_side[1]) * is_good[1]; */

            all += count;
        }
    }
    // printf ("prob = %lf\n", prob/all);
    if (all < 0.00000001) // all==0
    {
        return 0;
    }
    return time_average / all;
}

/* current_st is scratch space, indexed directly: it is filled in as the
 * recursion goes down and only the prefix of length current_index is
 * meaningful. */
static cofactoring_strategy gen_strat_r0_r1_ileav_st_rec(
    cofactoring_strategy const * strat_r0,
    unsigned int index_r0, cofactoring_strategy const * strat_r1, unsigned int index_r1,
    std::array<factorization_pattern_list, 2> const & init_tab, unsigned int * fbb,
    unsigned int * lpb, unsigned int * r, cofactoring_strategy & current_st,
    int current_index)
{
    unsigned int const len_r0 = strat_r0->steps.size();
    unsigned int const len_r1 = strat_r1->steps.size();

    /* if (index_r0 >= len_r0 && */
    /*     index_r1 >= len_r1) */
    /*   return nullptr; */
    /* printf ("%d (max= %d), %d (max=%d)\n", index_r0, len_r0, */
    /* 	  index_r1, len_r1); */

    std::optional<cofactoring_strategy> tmp0, tmp1;

    if (index_r0 < len_r0) {
        if (index_r0 < 5) {
            current_st.steps[current_index].method = strat_r0->steps[index_r0].method;
            current_st.steps[current_index].side = 0;
            tmp0 = gen_strat_r0_r1_ileav_st_rec(
                strat_r0, index_r0 + 1, strat_r1, index_r1, init_tab, fbb, lpb,
                r, current_st, current_index + 1);
        } else // concat the stay of our method!!!!
        {
            int const len = len_r0 - index_r0;
            for (int i = 0; i < len; i++) {
                current_st.steps[current_index + i].method = strat_r0->steps[index_r0 + i].method;
                current_st.steps[current_index + i].side = 0;
            }
            tmp0 = gen_strat_r0_r1_ileav_st_rec(
                strat_r0, len_r0, strat_r1, index_r1, init_tab, fbb, lpb, r,
                current_st, current_index + len);
        }
        // strategy_print (tmp0);
    }

    if (index_r1 < len_r1) {
        if (index_r1 < 5) {
            current_st.steps[current_index].method = strat_r1->steps[index_r1].method;
            current_st.steps[current_index].side = 1;
            tmp1 = gen_strat_r0_r1_ileav_st_rec(
                strat_r0, index_r0, strat_r1, index_r1 + 1, init_tab, fbb, lpb,
                r, current_st, current_index + 1);
        } else // concat the stay of our method!!!!
        {
            int const len = len_r1 - index_r1;
            for (int i = 0; i < len; i++) {
                current_st.steps[current_index + i].method = strat_r1->steps[index_r1 + i].method;
                current_st.steps[current_index + i].side = 1;
            }
            tmp1 = gen_strat_r0_r1_ileav_st_rec(
                strat_r0, index_r0, strat_r1, len_r1, init_tab, fbb, lpb, r,
                current_st, current_index + len);
        }

        // strategy_print (tmp1);
    }

    // end of the recursion!

    if (!tmp0 && !tmp1) {
        cofactoring_strategy final_st;
        // copy the current strategy!
        for (int i = 0; i < current_index; i++)
            final_st.add(current_st.steps[i].method, current_st.steps[i].side);
        double const prob = strat_r0->proba * strat_r1->proba;
        final_st.proba = prob;

        double const time =
            compute_time_strategy_ileav(init_tab, &final_st, fbb, lpb, r);
        final_st.time = time;
        /* printf ("final_st\n"); */
        /* strategy_print (final_st); */
        /* getchar (); */
        /* printf ("temps = %lf, index_current = %d\n", time, current_index); */
        return final_st;
    }

    if (!tmp0)
        return std::move(*tmp1);
    else if (!tmp1)
        return std::move(*tmp0);
    else if (tmp0->time < tmp1->time)
        return std::move(*tmp0);
    else //(tmp0->time > tmp1->time)
        return std::move(*tmp1);
}

static cofactoring_strategy gen_strat_r0_r1_ileav_st_rec(
    cofactoring_strategy const * strat_r0, cofactoring_strategy const * strat_r1,
    std::array<factorization_pattern_list, 2> const & init_tab,
    unsigned int * fbb, unsigned int * lpb, unsigned int * r)
{
    cofactoring_strategy scratch;
    scratch.steps.resize(strat_r0->steps.size() + strat_r1->steps.size());
    return gen_strat_r0_r1_ileav_st_rec(strat_r0, 0, strat_r1, 0, init_tab,
                                        fbb, lpb, r, scratch, 0);
}

static MAYBE_UNUSED cofactoring_strategy
gen_strat_r0_r1_ileav_st(cofactoring_strategy * strat_r0, cofactoring_strategy * strat_r1,
                         std::array<factorization_pattern_list, 2> const & init_tab,
                         unsigned int * fbb, unsigned int * lpb,
                         unsigned int * r)
{
    unsigned int const len_r0 = strat_r0->steps.size();
    unsigned int const len_r1 = strat_r1->steps.size();
    int const max_len = len_r0 + len_r1;

    cofactoring_strategy st;
    unsigned int index_r0 = 0, index_r1 = 0;
    int i = 0;
    int sequence = 2;
    while (i < max_len) {
        int const test = sequence % 2;
        if (index_r0 < len_r0 && (test == 0 || index_r1 >= len_r1)) {
            st.add(strat_r0->steps[index_r0++].method, 0);
            i++;
        } else if (index_r1 < len_r1) {
            st.add(strat_r1->steps[index_r1++].method, 1);
            i++;
        } else if (index_r1 == len_r1 && index_r0 == len_r0)
            break;
        sequence = (sequence - test) / 2;
    }
    double const prob = strat_r0->proba * strat_r1->proba;
    st.proba = prob;

    double const time = compute_time_strategy_ileav(init_tab, &st, fbb, lpb, r);
    st.time = time;

    return st;
}

static cofactoring_strategy_list
gen_strat_r0_r1_ileav(cofactoring_strategy_list const & strat_r0,
                      cofactoring_strategy_list const & strat_r1,
                      std::array<factorization_pattern_list, 2> const & init_tab,
                      unsigned int * fbb, unsigned int * lpb, unsigned int * r)
{
    cofactoring_strategy_list res;
    unsigned int const len0 = strat_r0.size();
    unsigned int const len1 = strat_r1.size();
    // printf ("r0 = %u, r1=%u\n", r[0], r[1]);
    // printf ("CLASSIC proba=%lf, time=%lf\n", res->tab[0]->proba,
    // res->tab[0]->time);
    //   strategy_print (res->tab[0]);
    for (unsigned int r0 = 0; r0 < len0; r0++)
        for (unsigned int r1 = 0; r1 < len1; r1++) {
            /* strategy_t* st1 = gen_strat_r0_r1_ileav_st(strat_r0->tab[r0], */
            /* 					   strat_r1->tab[r1], */
            /* 					   init_tab, fbb, lpb, r); */
            cofactoring_strategy st2 = gen_strat_r0_r1_ileav_st_rec(
                &strat_r0[r0], &strat_r1[r1], init_tab, fbb, lpb, r);
            // printf ("INTERL proba=%lf, time=%lf\n", st->proba, st->time);
            // strategy_print (st);
            res.push_back(std::move(st2));
            // Test entrelacement:
            /* int current_side = st2->steps[0].side; */
            /* int nb_chg = 0; */
            /* for (int i = 1; i < st2->len_side; i++) */
            /*   { */
            /*     if (st2->steps[i].side != current_side) */
            /*       nb_chg++; */
            /*     current_side = st2->steps[i].side; */
            /*   } */
            /* if (nb_chg > 1) */
            /*   { */
            /*     printf ("r0 = %u, r1=%u\n", r[0], r[1]); */
            /* printf ("CLASSI proba=%lf, time=%lf\n", st1->proba, st1->time);
             */
            /* printf ("INTERL proba=%lf, time=%lf\n", st2->proba, st2->time);
             */
            // getchar();
            //}
            // compare probabilities:
        }
    return res;
}

#ifdef COMPILE_DEAD_CODE
/*
  Test an interleaving with cado!!
 */
static MAYBE_UNUSED strategy_matrix  generate_matrix_cado_ileav(
    char const * pattern_dir, cofactoring_method_list methods,
    unsigned long lim0, unsigned int lpb0, unsigned int mfb0,
    unsigned long lim1, unsigned int lpb1, unsigned int mfb1)
{
    /* all the optimal strategies for each pair (r0, r1) */
    strategy_matrix matrix(mfb0 + 1);
    for (auto & row: matrix)
        row.resize(mfb1 + 1);

    unsigned int const fbb0 = ceil(log2((double)(lim0 + 1)));
    unsigned int const fbb1 = ceil(log2((double)(lim1 + 1)));

    /*
       For each r0 in [fbb0..mfb0], we precompute and strore the data
       for all strategies.  Whereas, for each r1, the values will be
       computed sequentially.
     */
    std::vector<cofactoring_strategy_list> data_rat(mfb0 + 1);

    unsigned int lim = 2 * fbb0 - 1;
    for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
        factorization_pattern_list patterns;
        if (r0 >= lim) {
            auto filename =
                fmt::format("{}/decomp_{}_{}", pattern_dir, lim0, r0);
            std::ifstream is(filename);
            if (!(is >> patterns)) {
                throw cado::error("Cannot read {}", filename);
            }
        }
        data_rat[r0] =
            generate_strategy_cado(methods, patterns, fbb0, lpb0, r0);
    }

    /*
       read good elements for r_2 in the array data_r1 and compute the
       data for each r_1. So :
     */
    lim = 2 * fbb1 - 1;
    for (unsigned int r1 = 0; r1 <= mfb1; r1++) {
        printf("r1 = %u\n", r1);
        factorization_pattern_list patterns;
        if (r1 >= lim) {
            auto filename =
                fmt::format("{}/decomp_{}_{}", pattern_dir, lim1, r1);
            std::ifstream is(filename);
            if (!(is >> patterns)) {
                throw cado::error("Cannot read {}", filename);
            }
        }

        cofactoring_strategy_list strat_r1 =
            generate_strategy_cado(methods, patterns, fbb1, lpb1, r1);

        for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
            if (r0 < 2 * fbb0 - 1 || r1 < 2 * fbb1 - 1) {
                matrix[r0][r1] =
                    generate_strategy_r0_r1(data_rat[r0], strat_r1);
            } else {
                factorization_pattern_list patterns_r0;
                auto filename = fmt::format("{}/decomp_{}_{}",
                                            pattern_dir, lim0, r0);
                std::ifstream is(filename);
                if (!(is >> patterns_r0)) {
                    throw cado::error("Cannot read {}", filename);
                }

                const std::array<factorization_pattern_list, 2> init_tab {patterns_r0,
                                                        patterns};
                unsigned int fbb[2] = {fbb0, fbb1};
                unsigned int lpb[2] = {lpb0, lpb1};
                unsigned int r[2] = {r0, r1};
                matrix[r0][r1] = gen_strat_r0_r1_ileav(data_rat[r0], strat_r1,
                                                       init_tab, fbb, lpb, r);
            }
        }
    }

    return matrix;
}
#endif /* COMPILE_DEAD_CODE */

static strategy_matrix
generate_matrix_ileav(std::string const & pattern_dir,
                      std::string const & name_directory_str, unsigned long lim0,
                      unsigned int lpb0, unsigned int mfb0, unsigned long lim1,
                      unsigned int lpb1, unsigned int mfb1)
{
    /* all the optimal strategies for each pair (r0, r1) */
    strategy_matrix matrix(mfb0 + 1);
    for (auto & row: matrix)
        row.resize(mfb1 + 1);

    unsigned int const fbb0 = ceil(log2((double)(lim0 + 1)));
    unsigned int const fbb1 = ceil(log2((double)(lim1 + 1)));

    /*
       For each r0 in [fbb0..mfb0], we precompute and strore the data
       for all strategies.  Whereas, for each r1, the values will be
       computed sequentially.
     */
    std::vector<cofactoring_strategy_list> data_rat(mfb0 + 1);

    unsigned int lim = 2 * fbb0 - 1;
    for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
        // get back the best strategies for r0!
        auto const name = fmt::format("{}/strategies{}_{}",
                                      name_directory_str, lim0, r0);
        std::ifstream is(name);
        if (!is || !(is >> data_rat[r0]))
            throw cado::error("Parser error: can't read the file '{}'", name);
    }

    /*
       read good elements for r_2 in the array data_r1 and compute the
       data for each r_1. So :
     */
    lim = 2 * fbb1 - 1;
    for (unsigned int r1 = 0; r1 <= mfb1; r1++) {
        printf("r1 = %u\n", r1);
        factorization_pattern_list patterns;
        if (r1 >= lim) {
            auto filename =
                fmt::format("{}/decomp_{}_{}", pattern_dir, lim1, r1);
            std::ifstream is(filename);

            if (!(is >> patterns)) {
                throw cado::error("Cannot read {}", filename);
            }
        }
        // get back the best strategies for r1!
        auto const name = fmt::format("{}/strategies{}_{}",
                                      name_directory_str, lim1, r1);
        cofactoring_strategy_list strat_r1;
        std::ifstream is2(name);
        if (!is2 || !(is2 >> strat_r1))
            throw cado::error("Parser error: can't read the file '{}'", name);

        for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
            if (r0 < 2 * fbb0 - 1 || r0 > 3 * fbb0 - 2 || r1 < 2 * fbb1 - 1 ||
                r1 > 3 * fbb1 - 2) {
                matrix[r0][r1] =
                    generate_strategy_r0_r1(data_rat[r0], strat_r1);
            } else {
                factorization_pattern_list patterns_r0;
                auto filename = fmt::format("{}/decomp_{}_{}",
                                            pattern_dir, lim0, r0);
                std::ifstream is(filename);
                if (!(is >> patterns_r0)) {
                    throw cado::error("Cannot read {}", filename);
                }

                const std::array<factorization_pattern_list, 2> init_tab {patterns_r0,
                                                        patterns};
                unsigned int fbb[2] = {fbb0, fbb1};
                unsigned int lpb[2] = {lpb0, lpb1};
                unsigned int r[2] = {r0, r1};
                matrix[r0][r1] = gen_strat_r0_r1_ileav(data_rat[r0], strat_r1,
                                                       init_tab, fbb, lpb, r);
            }
        }
    }

    return matrix;
}

/************************************************************************/
/*                  To bench our strategies                             */
/************************************************************************/

#ifdef COMPILE_DEAD_CODE
static unsigned int select_random_index_dec(double total_count, factorization_pattern_list const & t,
                                   gmp_randstate_ptr state)
{
    // 100000 to consider approximation of distribution
    double const alea = gmp_urandomm_ui(state, 10000);
    unsigned int i = 0;
    double bound = (t[0].count / total_count) * 10000;
    unsigned int const len = (int)t.size();
    ASSERT_ALWAYS(len);
    while (i < (len - 1) && (alea - bound) >= 1) {
        i++;
        bound += (t[i].count / total_count) * 10000;
    }
    return i;
}
#endif /* COMPILE_DEAD_CODE */

/*
  This function compute directly the probabity and the time to find a
  factor in a good decomposition in a cofactor of length r.

TODO: a function of the same name, yet behaving somewhat differently, exists in
tests/sieve/strategies/test_generate_strategies.cpp
 */
#ifdef COMPILE_DEAD_CODE
static MAYBE_UNUSED weighted_success bench_proba_time_st(
    gmp_randstate_t state, facul_strategy_oneside const & cofactoring_strategy,
    factorization_pattern_list const & init_tab, int r, MAYBE_UNUSED int lpb)
{
    size_t nb_test = 0, nb_success = 0;
    int const nb_test_max = 100000;
    double time = 0;

    double sum_dec = 0;
    for (auto const & D: init_tab)
        sum_dec += D.count;

    std::vector<cxx_mpz> f;

    if (sum_dec == 0)
        return { 0, 0.0, 0 };

    while (nb_test < nb_test_max) {
        /* N is composed by two prime factors, by the previous
         * function. Therefore, if a non trivial split was found,
         * then the status can not be FACUL_MAYBE.  */
        unsigned int const index = select_random_index_dec(sum_dec, init_tab, state);
        unsigned int const len_p = init_tab[index][0];

        cxx_mpz const N = generate_composite_integer(state, len_p, r);

        f.clear();
        time -= microseconds();
        auto const res = facul(N, cofactoring_strategy);
        time += microseconds();

        nb_success += res.status != FACUL_MAYBE;
        nb_test++;
    }

    return {nb_success, time, nb_test};
}
#endif /* COMPILE_DEAD_CODE */

#ifdef COMPILE_DEAD_CODE
// convert type: from strategy to facul_strategy_t.
static facul_strategy_oneside
convert_strategy_to_facul_strategy(cofactoring_strategy * t, unsigned long lim,
                                   unsigned int lpb, int side)
{
    std::vector<facul_method::parameters> mps;

    for (unsigned int i = 0; i < t->steps.size(); i++) {
        if (t->steps[i].side != side)
            continue;

        auto const & fm = t->steps[i].method;
        auto const method = fm.params.method;
        auto const curve = fm.params.parameterization;
        auto const B1 = fm.params.B1;
        auto const B2 = fm.params.B2;

        mps.emplace_back(method, B1, B2, curve, curve == MONTY16 ? 1UL : 4UL,
                         0 // extra_primes. It's 1 almost everywhere else, wtf?
        );
    }

    return { lim, lpb, 4 * lpb, mps, 0 };
}
#endif /* COMPILE_DEAD_CODE */

#ifdef COMPILE_DEAD_CODE
static facul_strategies convert_strategy_to_facul_strategies(
    cofactoring_strategy * t, const unsigned int * r, const unsigned long * fbb,
    const unsigned int * lpb, const unsigned int * mfb,
    gmp_randstate_ptr rstate)
{
    /* This is dead code, and it probably doesn't make sense to think
     * about strategies for >2 sides anyway.
     */
    std::vector<unsigned long> B(2);
    std::vector<unsigned int> lpb_(2);
    std::vector<unsigned int> mfb_(2);
    std::vector<unsigned int> r_(2);
    auto pB = B.begin();
    auto plpb = lpb_.begin();
    auto pmfb = mfb_.begin();
    auto * pr = r_.data();
    for (int side = 0; side < 2; side++) {
        *pB++ = *fbb++;
        *plpb++ = *lpb++;
        *pmfb++ = *mfb++;
        *pr++ = *r++;
    }

    std::vector<facul_method::parameters_with_side> mps;

    for (unsigned int i = 0; i < t->steps.size(); i++) {
        auto const & fm = t->steps[i].method;
        auto const method = fm.params.method;
        auto const curve = fm.params.parameterization;
        auto const B1 = fm.params.B1;
        auto const B2 = fm.params.B2;
        int const side = t->steps[i].side;
        unsigned long parameter = 1;
        if (method == EC_METHOD && curve != MONTY16) {
            for (; (parameter = u64_random(rstate)) < 2;)
                ;
        }
        mps.emplace_back(side, method, B1, B2, curve, parameter, 1);
    }

    facul_strategies::strategy_file S;
    S[r_] = mps;

    return {B, lpb_, mfb_, true, S, 0};
}
#endif /* COMPILE_DEAD_CODE */

#ifdef COMPILE_DEAD_CODE
static MAYBE_UNUSED weighted_success
bench_proba_time_st_both(gmp_randstate_t state, cofactoring_strategy * t,
                         std::array<factorization_pattern_list, 2> const & init_tab,
                         const unsigned int * r, const unsigned long * fbb,
                         const unsigned int * lpb, const unsigned int * mfb)
{
    /* This is dead code, and it probably doesn't make sense to think
     * about strategies for >2 sides anyway.
     */
    unsigned long lim[2] = {1UL << (fbb[0] - 1), 1UL << (fbb[1] - 1)};

    size_t nb_test = 0, nb_success = 0;
    size_t const nb_test_max = 10000;
    double time = 0;

    gmp_randstate_t state_copy;
    gmp_randinit_set(state_copy, state);

    double sum_dec[2] = {0, 0};
    for (int side = 0; side < 2; side++)
        for (auto const & D: init_tab[side])
            sum_dec[side] += D.count;

#if 1
    // Classic: bench without interleaving

    facul_strategy_oneside const facul_st_s0 =
        convert_strategy_to_facul_strategy(t, lim[0], lpb[0], 0);
    facul_strategy_oneside const facul_st_s1 =
        convert_strategy_to_facul_strategy(t, lim[1], lpb[1], 1);
    printf("classic\n");
    nb_success = 0;
    nb_test = 0;
    time = 0;
    {
        while (nb_test < nb_test_max) {
            cxx_mpz N[2];
            for (int side = 0; side < 2; side++) {
                int const index = select_random_index_dec(
                    sum_dec[side], init_tab[side], state);
                int const len_p = init_tab[side][index][1];
                N[side] = generate_composite_integer(state, len_p, r[side]);
            }
            /* N is composed by two prime factors, by the previous
             * function. Therefore, if a non trivial split was found,
             * then the status can not be FACUL_MAYBE.  */
            time -= microseconds();
            nb_success += facul(N[0], facul_st_s0).status != FACUL_MAYBE &&
                          facul(N[1], facul_st_s1).status != FACUL_MAYBE;
            time += microseconds();
            nb_test++;
            // getchar ();
        }
    }
    weighted_success res {nb_success, time, nb_test};
    printf("classic: prob = %lf, temps = %lf\n", res.prob, res.time);
#endif

#if 1
    printf("interl\n");
    facul_strategies const facul_st =
        convert_strategy_to_facul_strategies(t, r, lim, lpb, mfb, state);

    nb_success = 0;
    nb_test = 0;
    time = 0;

    {
        while (nb_test < nb_test_max) {
            std::vector<cxx_mpz> N(2);
            for (int side = 0; side < 2; side++) {
                int const index = select_random_index_dec(
                    sum_dec[side], init_tab[side], state);
                int const len_p = init_tab[side][index][1];
                N[side] =
                    generate_composite_integer(state_copy, len_p, r[side]);
            }
            /* N is composed by two prime factors, by the previous
             * function. Therefore, if a non trivial split was found,
             * then the status can not be FACUL_MAYBE.  */
            time -= microseconds();
            auto const res = facul_all(N, facul_st);
            nb_success +=
                res[0].status == FACUL_SMOOTH && res[1].status == FACUL_SMOOTH;
            time += microseconds();
            nb_test++;
            /* printf ("[%d, %d]\n", (int)mpz_sizeinbase (N[nb_test][0], 2), */
            /* 	  (int)mpz_sizeinbase (N[nb_test][1], 2)); */
            /* printf ("nb_success = %d\n", nb_success); */
            // getchar();
        }
    }
    weighted_success const res2 {nb_success, time, nb_test};
    printf("interL: proba = %lf, time = %lf\n", res2.prob, res2.time);
#endif

    gmp_randclear(state_copy);

    return res;
}
#endif /* COMPILE_DEAD_CODE */

/************************************************************************/
/*                            USAGE                                     */
/************************************************************************/

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header("This binary allows to test the cofactoring_strategy of cado, "
                            "and especially compute the theoretical number of "
                            "relations found per second by this cofactoring_strategy.\n");

    pl.declare_usage("lim0",
                          "set rational factor base bound to lim0\n");
    pl.declare_usage("lim1",
                          "set algebraic factor base bound to lim1\n");
    pl.declare_usage("lpb0",
                          "set rational large prime bound to 2^lpb0");
    pl.declare_usage("lpb1",
                          "set algebraic large prime bound to 2^lpb1");
    pl.declare_usage("mfb0", "set the first cofactor bound to 2^mfb0");
    pl.declare_usage("mfb1",
                          "set the second cofactor bound to 2^mfb1");
    pl.declare_usage("decomp",
        "to locate the file or the directory , according to\n"
        "\t \t if you need one or several files,\n"
        "\t \t which contain(s) the file(s) of cofactors decompositions.");
    pl.declare_usage("dist",
        "the pathname of our file which contains the distribution\n"
        "\t\t of our pairs of cofactors.");
    pl.declare_usage("t",
        "specify the time (seconds) to obtain cofactors in the file\n"
        "\t\t given by the option 'dist'.");
    pl.declare_usage("out",
                          "the output file which contain our strategies\n");
    pl.declare_usage("fm",
        "the file of benched factoring methods that describes cado's\n"
        "\t\t own cofactorization cofactoring_strategy (see bench_cofactoring_methods).");
    pl.declare_usage("in",
        "the directory of one-sided strategies precomputed by\n"
        "\t\t build_cofactoring_strategies -oneside.");
}

/************************************************************************/
/*     MAIN                                                             */
/************************************************************************/

static int main_(int argc, char const * argv[]);

// coverity[root_function]
int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;
    declare_usage(pl);

    pl.process_command_line_and_extra_parameter_files(argc, argv);

    // option parser
    unsigned long lim0 = 0;
    unsigned int lpb0 = 0;
    unsigned long lim1 = 0;
    unsigned int lpb1 = 0;
    unsigned int mfb0 = 0;
    unsigned int mfb1 = 0;
    double C0 = -1;

    pl.parse("lim0", lim0);
    pl.parse("lpb1", lpb1);
    pl.parse("lim1", lim1);
    pl.parse("mfb1", mfb1);
    pl.parse("lpb0", lpb0);
    pl.parse("mfb0", mfb0);
    pl.parse("t", C0);

    if (lim0 == 0 || lpb0 == 0 || mfb0 == 0 || lim1 == 0 || lpb1 == 0 ||
        mfb1 == 0 || C0 == -1) {
        fputs("ALL parameters are mandatory!\n", stderr);
        exit(EXIT_FAILURE);
    }


    // int fbb0 = ceil (log2 ((double) (lim0+1)));
    // int fbb1 = ceil (log2 ((double) (lim1+1)));
    // int fbb = (fbb0 < fbb1) ? fbb0 : fbb1;
    //     int lpb = (lpb0 > lpb1) ? lpb0 : lpb1;
    // convert the time in micro-s. because all previous binaries
    // compute their times in micro-s.
    C0 *= 1000000; // s-->micro-s

    auto const pattern_dir = pl.parse_mandatory<std::string>("decomp");
    auto const name_file_cofactor = pl.parse_mandatory<std::string>("dist");

    std::ifstream file_C(name_file_cofactor);
    if (!file_C)
        throw cado::error("Error while reading file {}", name_file_cofactor);
    auto const distrib_C = extract_matrix_C(file_C, mfb0 + 1, mfb1 + 1);

    cxx_gmp_randstate state;

    auto const name_file_fm = pl.parse_mandatory<std::string>("fm");
    cofactoring_method_list methods;
    {
        std::ifstream file_in(name_file_fm);
        if (!file_in || !(file_in >> methods))
            throw cado::error("Cannot read {}", name_file_fm);
    }
    /* we set mfb = 3*lpb0 to avoid the special-case of 2 large primes */
    fmt::print("len  = {}, ({})\n", methods.size(),
               3 + nb_curves(lpb0, 3 * lpb0));

    // test computation of probabilities
    //{tab_init
    /* int fbb0 = ceil (log2 ((double) (lim0+1))); */
    /* int fbb1 = ceil (log2 ((double) (lim1+1))); */

    /* int r0 = 62, r1=63; */
    /* assert (r0 <= mfb0 && r1 <= mfb1); */
    /* printf ("r0 = %u, r1= %u\n", r0, r1); */
    /* char name_file[200]; */
    /* snprintf(name_file, sizeof(name_file), */
    /* 	    "%s/decomp_%lu_%u", pattern_dir, lim0, r0);//modify it!!
     */
    /* printf ("%s\n", name_file); */
    /* fflush(stdout); */
    /* FILE *file = fopen(name_file, "r"); */
    /* tabular_decomp_t* init_tab0 = tabular_decomp_fscan (file); */
    /* fclose (file); */
    /* snprintf(name_file, sizeof(name_file), */
    /* 	    "%s/decomp_%lu_%u", pattern_dir, lim1, r1);//modify it!!
     */
    /* printf ("%s\n", name_file); */
    /* file = fopen(name_file, "r"); */
    /* tabular_decomp_t* init_tab1 = tabular_decomp_fscan (file); */
    /* fclose (file); */
    /* //Test{{{ */
    /* /\* init_tab0->index = 1; *\/ */
    /* /\* init_tab1->index = 1; *\/ */
    /* /\* tabular_decomp_print (init_tab0); *\/ */
    /* //}}} */
    /* //  } */
    /* strategy* strat1 = strategy_create(); */
    /* for (int i = 0; i < methods->index; i++) */
    /*   strategy_add_fm (strat1, methods->tab[i]); */
    /* /\* strategy_add_fm (strat1, methods->tab[0]); *\/ */
    /* /\* strategy_add_fm (strat1, methods->tab[1]); *\/ */
    /* /\* strategy_add_fm (strat1, methods->tab[2]); *\/ */
    /* /\* strategy_add_fm (strat1, methods->tab[3]); *\/ */
    /* /\* strategy_add_fm (strat1, methods->tab[4]); *\/ */
    /* /\* strategy_add_fm (strat1, methods->tab[5]); *\/ */
    /* /\* strategy_add_fm (strat1, methods->tab[6]); *\/ */
    /* strategy* strat2 = strategy_create(); */
    /* for (int i = 0; i < methods->index; i++) */
    /*   strategy_add_fm (strat2, methods->tab[i]); */

    /* /\* strategy_add_fm (strat2, methods->tab[0]); *\/ */
    /* /\* strategy_add_fm (strat2, methods->tab[1]); *\/ */
    /* /\* strategy_add_fm (strat2, methods->tab[2]); *\/ */
    /* /\* strategy_add_fm (strat2, methods->tab[3]); *\/ */
    /* /\* strategy_add_fm (strat2, methods->tab[4]); *\/ */
    /* /\* strategy_add_fm (strat2, methods->tab[5]); *\/ */
    /* /\* strategy_add_fm (strat2, methods->tab[6]); *\/ */
    /* /\* strategy_print (strat1); *\/ */
    /* /\* strategy_print (strat2); *\/ */
    /* double proba1 = compute_proba_strategy (init_tab0, strat1, fbb0, lpb0);
     */
    /* double proba2 = compute_proba_strategy (init_tab1, strat2, fbb1, lpb1);
     */
    /* double temps1 = compute_time_strategy (init_tab0, strat1, r0); */
    /* double temps2 = compute_time_strategy (init_tab1, strat2, r1); */
    /* strategy_set_proba (strat1, proba1); */
    /* strategy_set_proba (strat2, proba2); */
    /* strategy_set_time (strat1, temps1); */
    /* strategy_set_time (strat2, temps2); */
    /* printf ("proba1 = %lf, proba2 = %lf, combo = %lf\n", */
    /* 	    proba1, proba2, proba1*proba2); */
    /* printf ("temps1 = %lf, temps2 = %lf, combo = %lf, %lf\n", */
    /* 	    temps1, temps2, temps1 + proba1*temps2, temps2 + proba2*temps1); */
    /* int fbb[2] = {fbb0, fbb1}; */
    /* int lpb[2] ={lpb0, lpb1}; */
    /* int r[2] = {r0, r1}; */
    /* tabular_decomp_t* init_tab[2] = {init_tab0, init_tab1}; */
    /* //{ */
    /* strat1->len_side = 3; */
    /* strat1->side = malloc (sizeof (int) * strat1->len_side); */
    /* strat1->steps[0].side = strat1->steps[1].side = strat1->steps[2].side =0; */
    /* //  } */
    /* /\* double temps = compute_time_strategy_ileav(init_tab, strat1, fbb,
     * lpb, r); *\/ */
    /* /\* printf ("temps %lf\n", temps); *\/ */
    /* /\* strategy *res = gen_strat_r0_r1_ileav_st(strat1, strat2, init_tab,
     * *\/ */
    /* /\* 					       fbb, lpb, r); *\/ */
    /* strategy* res = gen_strat_r0_r1_ileav_st_rec(strat1, 0, */
    /* 						   strat2, 0, */
    /* 						   init_tab, fbb, lpb, r, */
    /* 						   nullptr, 0); */
    /* printf ("resutat!!!!\n"); */
    /* printf ("theo: proba = %lf, tps = %lf\n", res->proba, res->time); */
    /* //strategy_print (res); */
    /* //bench time with our interleaving methods! */
    /* int mfb[2] = {mfb0, mfb1}; */
    /* printf ("bench proba time\n"); */
    /* double* tmp = bench_proba_time_st_both(state, res, init_tab, r, fbb, lpb,
     * mfb); */
    /* printf ("bench both interL: proba = %lf, time = %lf\n", tmp[0], tmp[1]);
     */
    /* exit(1); */
    //}}

    // generate our strategies
    // remark: for each pair (r0, r1), we have only one strategy!!
#ifndef CADO_INTERLEAVING
    // clear me!!!!
    // compute our strategy
    strategy_matrix  matrix_strat;
    if (0) // my classic matrix!
    {
        char const * pathname_st = "../res_matrix";
        matrix_strat = extract_matrix_strat(pathname_st, mfb0 + 1, mfb1 + 1);
    } else // compute interleaving strategies!
    {
        auto const name_directory_str = pl.parse_mandatory<std::string>("in");
        matrix_strat =
            generate_matrix_ileav(pattern_dir, name_directory_str,
                                  lim0, lpb0, mfb0, lim1, lpb1, mfb1);
    }
    printf("our strategy_file!\n");
    auto const matrix_strat_res =
        compute_best_strategy(matrix_strat, distrib_C, mfb0 + 1, mfb1 + 1, C0);

    /* exit(1); */
    //}}

    strategy_matrix const matrix = generate_matrix_cado(
        pattern_dir, methods, lim0, lpb0, mfb0, lim1, lpb1, mfb1);

    // eval our strategy!
    double Y = 0, T = C0;
    for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
        for (unsigned int r1 = 0; r1 <= mfb1; r1++) {
            Y += double(distrib_C[r0][r1]) * matrix[r0][r1][0].proba;
            T += double(distrib_C[r0][r1]) * matrix[r0][r1][0].time;
            // test: Add test with our strategy!
            if (0) // matrix_strat_res[r0][r1] != nullptr)
            {
                fmt::print("cado r0={}, r1={}, p ={:f}, t = {:f}\n", r0, r1,
                       matrix[r0][r1][0].proba,
                       matrix[r0][r1][0].time);
                fmt::print("file r0={}, r1={}, p ={:f}, t = {:f}\n", r0, r1,
                       matrix_strat_res[r0][r1]->proba,
                       matrix_strat_res[r0][r1]->time);
                printf("number of pair: %lu\n", distrib_C[r0][r1]);
                /* 	strategy_print (matrix_strat_res[r0][r1]); */
                /* //getchar(); */
            }
            // end of test
        }
    }
    // print the result!
    printf("without interleaving with cado!\n");
    printf(" Y = %lf relations, T = %lf s., yt = %1.10lf s/rel\n", Y,
           T / 1000000, T / (Y * 1000000));

    auto const pathname_output = pl.parse_mandatory<std::string>("out");
    std::ofstream file_output(pathname_output);
    if (!file_output)
        throw cado::error("Cannot write {}", pathname_output);
    fprint_final_strategy(file_output, matrix_strat_res, mfb0 + 1, mfb1 + 1);
    /* if (pathname_output != nullptr) { */
    /* 	FILE *file_output = fopen(pathname_output, "w"); */
    /* 	for (unsigned int r0 = 0; r0 <= mfb0; r0++) */
    /* 	    for (unsigned int r1 = 0; r1 <= mfb1; r1++) */
    /* 		if (matrix[r0][r1]->tab[0] != nullptr) { */
    /* 		    fprintf(file_output, */
    /* 			    "[r0=%u, r1=%u] : (p = %lf, t = %lf)\n", */
    /* 			    r0, r1, matrix[r0][r1]->tab[0]->proba, */
    /* 			    matrix[r0][r1]->tab[0]->time); */
    /* 		    strategy_fprint_design(file_output, matrix[r0][r1]->tab[0]);
     */
    /* 		} */
    /* 	fclose(file_output); */
    /* } */
#else // interleaving!

    strategy_matrix  matrix = generate_matrix_cado_ileav(
        pattern_dir, methods, lim0, lpb0, mfb0, lim1, lpb1, mfb1);

    cofactoring_strategy *** matrix_strat_res =
        compute_best_strategy(matrix, distrib_C, mfb0 + 1, mfb1 + 1, C0);

    char const * pathname_output;
    pathname_output = pl.lookup_old("out");

    if (pathname_output != nullptr) {
        FILE * file_output = fopen(pathname_output, "w");
        fprint_final_strategy(file_output, matrix_strat_res, mfb0 + 1,
                              mfb1 + 1);
        fclose(file_output);
    }
    // free matrix_strat_res
    for (unsigned int r0 = 0; r0 <= mfb0; r0++) {
        for (unsigned int r1 = 0; r1 <= mfb1; r1++)
            strategy_free(matrix_strat_res[r0][r1]);
        free(matrix_strat_res[r0]);
    }
    free(matrix_strat_res);
#endif

    //{{test bench strategy!
    /* int r0 = CONST_TEST_R; */
    /* int r1 = CONST_TEST_R; */
    /* printf ("r0 = %u, r1 = %u, lpb0 = %u, lpb1 = %u\n", r0, r1, lpb0, lpb1);
     */
    /* //printf ("nb call = %lu\n", distrib_C[r0][r1]); */
    /* printf ("proba found: %lf, %lf\n", matrix[r0][r1]->tab[0]->proba,
     * matrix[r0][r1]->tab[0]->time); */
    /* char name_file[200]; */
    /* snprintf(name_file, sizeof(name_file), */
    /* 	    "%s/decomp_%u_%u", pattern_dir, fbb0, r0); */
    /* FILE *file = fopen(name_file, "r"); */

    /* tabular_decomp_t* patterns = tabular_decomp_fscan(file); */

    /* tabular_decomp_print (patterns); */
    /* fclose (file); */

    /* facul_strategy_t* facul_st = facul_make_strategy (fbb0, lpb0, 0, 0); */

    /* double* res = bench_proba_time_st(state, facul_st, patterns, r0, lpb0);
     */
    /* double p0 = res[0], t0 = res[1]; */
    /* printf ("side = 0, proba = %lf, time = %lf\n", res[0], res[1]); */
    /* tabular_decomp_free (patterns); */
    /* facul_clear_strategy(facul_st); */
    /* //r1 */
    /* snprintf(name_file, sizeof(name_file), */
    /* 	    "%s/decomp_%u_%u", pattern_dir, fbb1, r1); */
    /* file = fopen(name_file, "r"); */

    /* patterns = tabular_decomp_fscan(file); */
    /* fclose (file); */

    /* facul_st = facul_make_strategy (fbb1, lpb1, 0, 0); */
    /* res = bench_proba_time_st(state, facul_st, patterns, r1, lpb1); */
    /* printf ("side = 1, proba = %lf, time = %lf\n", res[0], res[1]); */
    /* printf ("two sides, proba = %lf, time = %lf\n", res[0]*p0, t0+p0*res[1]);
     */
    /* tabular_decomp_free (patterns); */
    /* //Free facil_st */
    /* facul_clear_strategy(facul_st); */
    //}}

    return EXIT_SUCCESS;
}
