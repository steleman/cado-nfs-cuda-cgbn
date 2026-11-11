#include "cado.h" // IWYU pragma: keep

#include <cstdlib>

#include <set>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "ecm/stage2.h"
#include "getprime.h"

/* stage2_make_plan() chooses a parameter w and emits the (u, v) pairs whose
 * products cover the primes in ]B1, B2]. Two things are checked here for
 * each pair of bounds:
 *
 *  - that a plan comes out at all. The choice of w used to abort the
 *    process whenever B2 landed just below B1^2, which is reachable from
 *    the command line of the tools in sieve/strategies;
 *
 *  - that the plan covers every prime in ]B1, B2]. That is the property the
 *    rest of stage 2 relies on, and it is what says that giving up the
 *    B1^2 bound on w -- which is only about the cost of the plan -- leaves
 *    a plan that still works.
 */

/* The primes in ]B1, B2] that the plan covers.
 *
 * ecm_stage2() forms the product of the X_{v w P} - X_{u P} over the plan's
 * pairs, which is divisible by p exactly when p divides v*w-u or v*w+u. So
 * a pair covers every prime factor of those two values that lies in the
 * interval, not just the values themselves.
 *
 * The pairs are walked the way ecm_stage2() walks them: v starts at vmin
 * and goes up by one on every PAIR_INCR_V, and each other entry is an index
 * into U.
 */
static std::set<unsigned int> covered_primes(stage2_plan_t const & plan,
                                             unsigned int B1, unsigned int B2)
{
    std::set<unsigned int> values;
    unsigned int v = plan.vmin;
    for (unsigned int const * p = plan.pairs; *p != PAIR_END; p++) {
        if (*p == PAIR_INCR_V) {
            v++;
            continue;
        }
        unsigned int const u = plan.U[*p];
        values.insert(v * plan.w + u);
        if (v * plan.w > u)
            values.insert(v * plan.w - u);
    }

    std::set<unsigned int> covered;
    for (unsigned int const n: values) {
        unsigned int m = n;
        for (unsigned int d = 2; (unsigned long) d * d <= m; d++)
            for (; m % d == 0; m /= d)
                if (d > B1 && d <= B2)
                    covered.insert(d);
        if (m > B1 && m <= B2)
            covered.insert(m);
    }
    return covered;
}

static bool check(unsigned int B1, unsigned int B2,
                  stage2_cost_t const & opcost, std::string const & what)
{
    stage2_plan_t plan {};
    stage2_make_plan(&plan, B1, B2, &opcost, 0);

    bool ok = true;
    auto fail = [&](std::string const & why) {
        fmt::print(stderr, "error: {}, B1={}, B2={}: {}\n", what, B1, B2, why);
        ok = false;
    };

    if (B2 <= B1 || B1 < 3) {
        /* Outside the domain: there is no w to build a plan with. What
         * matters is that saying so is not fatal. */
        if (plan.w != 0)
            fail(fmt::format("expected no plan, got w={}", plan.w));
    } else {
        std::vector<unsigned int> primes;
        prime_info pi;
        prime_info_init(pi);
        unsigned int p;
        for (p = 2; p <= B1; p = (unsigned int) getprime_mt(pi))
            ;
        /* p is now the smallest prime > B1 */
        for (; p <= B2; p = (unsigned int) getprime_mt(pi))
            primes.push_back(p);
        prime_info_clear(pi);

        if (plan.w == 0 || plan.w % 6) {
            fail(fmt::format("w={} is not a usable plan", plan.w));
        } else {
            auto const covered = covered_primes(plan, B1, B2);
            for (unsigned int const q: primes)
                if (!covered.count(q)) {
                    fail(fmt::format("prime {} is not covered (w={})", q,
                                     plan.w));
                    break;
                }
        }
    }

    stage2_clear_plan(&plan);
    return ok;
}

int main()
{
    stage2_cost_t const ecm_cost {1, 5.0, 6.0};
    stage2_cost_t const pm1_cost {0, 1.0, 1.0};

    struct bounds { unsigned int B1, B2; };

    bool ok = true;

    /* B2 just below B1^2 is what used to abort: no w is both a multiple of
     * 6 below 6*B1 and below B1^2-B2, so the search came up empty. */
    for (unsigned int const B1: {10U, 20U, 50U, 97U}) {
        for (unsigned int const d: {0U, 1U, 6U, 29U, 30U, 31U}) {
            unsigned int const B2 = B1 * B1 - d;
            ok &= check(B1, B2, ecm_cost, "B2 just below B1^2");
            ok &= check(B1, B2, pm1_cost, "B2 just below B1^2, P-1");
        }
        /* and above it, where B1*B1-B2 used to wrap around */
        ok &= check(B1, B1 * B1 + 1, ecm_cost, "B2 just above B1^2");
        ok &= check(B1, 4 * B1 * B1, ecm_cost, "B2 well above B1^2");
    }

    /* the sweep create_strat_file.sh runs, which reaches c = B1 */
    for (unsigned int c = 10; c <= 50; c += 5)
        for (unsigned int B1 = 50; B1 <= 300; B1 += 20)
            ok &= check(B1, B1 * c, pm1_cost, "create_strat_file.sh sweep");

    /* ordinary bounds, including what facul's default strategy asks for */
    bounds const ordinary[] = {{315, 2205}, {525, 3255}, {105, 3255},
                               {315, 5355}, {3000, 150000}};
    for (auto const & b: ordinary)
        ok &= check(b.B1, b.B2, ecm_cost, "default strategy");

    /* bounds outside the domain must not be fatal either */
    bounds const degenerate[] = {{0, 0},   {0, 100},   {1, 100},
                                {2, 100}, {100, 100}, {100, 50}};
    for (auto const & b: degenerate)
        ok &= check(b.B1, b.B2, ecm_cost, "outside the domain");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
