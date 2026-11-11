#include "cado.h" // IWYU pragma: keep

#include <cstdlib>

#include <sstream>
#include <string>
#include <vector>

#include "fmt/base.h"

#include "cofactoring_method.hpp"
#include "cofactoring_strategy.hpp"

static cofactoring_method make(unsigned long method, unsigned long curve,
                             unsigned long B1, unsigned long B2)
{
    cofactoring_method fm = cofactoring_method::from_fields(method, curve, B1, B2);
    fm.len_p_min = 20;
    fm.proba = {0.5};
    fm.time = {1, 2, 3, 4};
    return fm;
}

/* The file format keeps only the method, the curve, the bounds and the
 * side, so that is all a round trip can preserve. */
static bool same_chain(cofactoring_strategy const & a,
                       cofactoring_strategy const & b)
{
    if (a.steps.size() != b.steps.size())
        return false;
    for (size_t i = 0; i < a.steps.size(); i++) {
        if (!a.steps[i].method.same_method_as(b.steps[i].method))
            return false;
        if (a.steps[i].side != b.steps[i].side)
            return false;
    }
    return true;
}

int main()
{
    int rc = EXIT_SUCCESS;
    auto fail = [&rc](std::string const & what) {
        fmt::print(stderr, "error: {}\n", what);
        rc = EXIT_FAILURE;
    };

    /* a step added without a side sits on side 0 */
    {
        cofactoring_strategy s;
        s.add(make(PM1_METHOD, 0, 315, 2205));
        s.add(make(EC_METHOD, MONTY12, 105, 3255), 1);
        if (s.steps[0].side != 0)
            fail("a step added without a side did not default to side 0");
        if (s.steps[1].side != 1)
            fail("the side of a step came back wrong");
    }

    /* a step converts to what las ends up holding */
    {
        cofactoring_strategy s;
        s.add(make(EC_METHOD, MONTY12, 105, 3255), 1);
        auto const p = s.steps[0].parameters();
        if (p.side != 1 || p.method != EC_METHOD || p.B1 != 105 ||
            p.B2 != 3255 || p.parameterization != MONTY12)
            fail("step::parameters() lost something on the way");
    }

    /* round trip through the stream operators */
    {
        cofactoring_strategy_list t;

        cofactoring_strategy s1;
        s1.add(make(PM1_METHOD, 0, 315, 2205), 0);
        s1.add(make(EC_METHOD, MONTY12, 105, 3255), 1);
        s1.proba = 0.75;
        s1.time = 123.5;
        t.push_back(s1);

        cofactoring_strategy s2;
        s2.add(make(EC_METHOD, MONTY16, 0, 0), 1);
        s2.proba = 0;
        s2.time = 0;
        t.push_back(s2);

        std::ostringstream os;
        os << t;

        cofactoring_strategy_list back;
        std::istringstream is(os.str());
        if (!(is >> back))
            fail("could not read back what we just wrote");
        if (back.size() != t.size()) {
            fail("the round trip changed the number of strategies");
        } else {
            for (size_t i = 0; i < t.size(); i++) {
                if (!same_chain(t[i], back[i]))
                    fail("the round trip changed a chain of methods");
                /* Probability is written with ten decimals, time with
                 * six; both of these survive exactly. */
                if (back[i].proba != t[i].proba)
                    fail("the round trip changed a probability");
                if (back[i].time != t[i].time)
                    fail("the round trip changed a time");
            }
        }
    }

    /* comments are skipped, so that a file with the section labels that
     * the pinned test fixtures carry is still valid input */
    {
        std::istringstream is("# strategies1024_21\n"
                              "PM1 0 0 0\n"
                              "Probability: 0.0000000000\n"
                              "Time: 0.000000\n"
                              "# strategies1024_22\n"
                              "PM1 100 1000 0\n"
                              "Probability: 0.5000000000\n"
                              "Time: 1.000000\n");
        cofactoring_strategy_list t;
        if (!(is >> t) || t.size() != 2)
            fail("a commented, labelled strategy file did not parse");
    }

    /* likewise, a strategy file in the old numeric format is refused */
    {
        std::istringstream is("1 2 100 1000 0\n"
                              "Probability: 0.5000000000\n"
                              "Time: 1.000000\n");
        cofactoring_strategy_list t;
        if (is >> t)
            fail("a strategy file in the old numeric format was accepted");
    }

    /* malformed input fails the stream rather than killing the process */
    {
        std::istringstream is("PM1 315 2205 0\nProbability: oops\n");
        cofactoring_strategy_list t;
        if (is >> t)
            fail("a malformed cofactoring_strategy was accepted");
    }

    /* an empty input is an empty table, not a failure */
    {
        std::istringstream is("");
        cofactoring_strategy_list t;
        if (!(is >> t) || !t.empty())
            fail("an empty input was not read as an empty table");
    }

    return rc;
}
