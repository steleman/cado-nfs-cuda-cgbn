#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <exception>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "ecm/facul_ecm.h"
#include "ecm/facul_method.hpp"
#include "ecm/facul_strategies.hpp"
#include "macros.h"
#include "utils_cxx.hpp"

/* The strategy file is written by choose_cofactoring_strategies, in sieve/strategies, and read
 * back here. Until now only the reading half was tested. This writes one
 * method of every family the parser knows about, reads the result back,
 * and checks that nothing changed on the way.
 */

/* One method of every family the strategy file format names, taken from
 * facul_method_families() rather than listed again here, so that a family
 * added there is covered without touching this test. */
static std::vector<facul_method::parameters_with_side> every_family()
{
    std::vector<facul_method::parameters_with_side> v;
    int side = 0;
    for (auto const & f: facul_method_families()) {
        /* the second curve of the sequence, which a bare name in
         * first position would not give: the parameter must be written */
        if (f.method == EC_METHOD)
            v.emplace_back(side, EC_METHOD, 105UL, 3255UL, f.parameterization,
                           ec_valid_parameter_from_sequence(f.parameterization, 1), 1);
        else
            v.emplace_back(side, f.method, 315UL, 2205UL);
        side = 1 - side;
    }
    return v;
}

int main()
{
    int rc = EXIT_SUCCESS;
    auto fail = [&rc](std::string const & what) {
        fmt::print(stderr, "error: {}\n", what);
        rc = EXIT_FAILURE;
    };

    /* What choose_cofactoring_strategies produces is a strategy_file:
     * a chain of methods for each pair of cofactor sizes. Build one,
     * write it the way that binary does, and read it back through the
     * parser las uses. */
    facul_strategies::strategy_file wanted;
    wanted[{30, 30}] = every_family();
    wanted[{28, 31}] = {
        facul_method::parameters_with_side(0, PM1_METHOD, 315UL, 2205UL),
        facul_method::parameters_with_side(1, EC_METHOD, 105UL, 3255UL,
                                           MONTYTWED12, 2UL, 1),
    };
    /* a STOP marker after the P-1 of side 1 */
    wanted[{29, 40}] = {
        facul_method::parameters_with_side(1, PM1_METHOD, 315UL, 2205UL),
        facul_method::parameters_with_side(0, PM1_METHOD, 315UL, 2205UL),
        facul_method::parameters_with_side(1, EC_METHOD, 105UL, 3255UL,
                                           MONTYTWED12, 2UL, 1),
    };
    wanted[{29, 40}][0].stop_unsplit = true;
    /* MPQS, which is not a family */
    wanted[{29, 41}] = {
        facul_method::parameters_with_side(1, PM1_METHOD, 315UL, 2205UL),
        facul_method::parameters_with_side(1, MPQS_METHOD, 0UL, 0UL),
    };

    /* every family must have a name, and they must all differ */
    auto const & families = wanted[{30, 30}];
    for (size_t i = 0; i < families.size(); i++)
        for (size_t j = i + 1; j < families.size(); j++)
            if (facul_method_name(families[i]) == facul_method_name(families[j]))
                fail(fmt::format("two families share the name {}",
                                 facul_method_name(families[i])));

    /* facul_method_by_name() is the inverse of facul_method_name() */
    for (auto const & f: facul_method_families()) {
        auto const name = facul_method_name(f);
        facul_method::parameters back;
        if (!facul_method_by_name(name, back)) {
            fail(fmt::format("{} is a family but is not recognised by name",
                             name));
            continue;
        }
        if (facul_method_name(back) != name)
            fail(fmt::format("{} came back as {}", name,
                             facul_method_name(back)));
    }

    /* and it rejects what is not a family. ECM-TM16 is the interesting
     * one: parameterization_name() has a name for MONTYTWED16, but
     * nothing can run it, so no reader may accept it. */
    for (auto const & name: {"ECM-TM16", "ECM-M13", "", "PM2"}) {
        facul_method::parameters back;
        if (facul_method_by_name(name, back))
            fail(fmt::format("'{}' was accepted as a family name", name));
    }

    /* Nothing in the list may be a name that kills the process further
     * down: every ECM family must have a curve construction, and must be
     * able to hand out a parameter. This is what keeps MONTYTWED16 out. */
    for (auto const & f: facul_method_families()) {
        if (f.method != EC_METHOD)
            continue;
        if (!ec_parameterization_is_implemented(f.parameterization)) {
            fail(fmt::format("{} is offered but has no curve construction",
                             facul_method_name(f)));
            continue;
        }
        auto const param = ec_valid_parameter_from_sequence(f.parameterization, 0);
        if (!ec_parameter_is_valid(f.parameterization, param))
            fail(fmt::format("{}: parameter {} from the sequence is not valid",
                             facul_method_name(f), param));
    }

    /* The default parameter sequence must hand out usable, distinct
     * curves: every value valid, and none repeated. A repeat would mean
     * a chain running the same curve twice while the model charges it
     * for two independent attempts. */
    for (auto const & f: facul_method_families()) {
        if (f.method != EC_METHOD)
            continue;
        auto const para = f.parameterization;
        std::set<unsigned long> seen;
        for (unsigned long k = 0; k < 32; k++) {
            auto const param = ec_valid_parameter_from_sequence(para, k);
            if (!ec_parameter_is_valid(para, param)) {
                fail(fmt::format("{}: sequence value {} is the singular"
                                 " parameter {}",
                                 facul_method_name(f), k, param));
                break;
            }
            /* MONTY16 is one single curve, so it is allowed to repeat */
            if (para != MONTY16 && !seen.insert(param).second) {
                fail(fmt::format("{}: parameter {} is handed out twice by"
                                 " the sequence, at index {}",
                                 facul_method_name(f), param, k));
                break;
            }
        }
    }

    /* the usage strings are built from the same list */
    for (auto const & f: facul_method_families())
        if (facul_method_family_names().find(facul_method_name(f)) ==
            std::string::npos)
            fail(fmt::format("{} is missing from facul_method_family_names()",
                             facul_method_name(f)));

    std::ostringstream os;
    facul_strategies::write_file(os, wanted);
    std::string const text = os.str();

    // coverity[secure_temp]
    FILE * f = tmpfile();
    DIE_ERRNO_DIAG(f == nullptr, "tmpfile(%s)", "");
    fputs(text.c_str(), f);
    fflush(f);

    std::vector<unsigned long> const B(2, 1UL << 20);
    std::vector<unsigned int> const lpb(2, 25);
    std::vector<unsigned int> const mfb(2, 50);

    try {
        auto const F = facul_strategies(B, lpb, mfb, true, f, 0);
        for (auto const & [sizes, chain]: wanted) {
            auto const & got = F(sizes);
            if (got.size() != chain.size()) {
                fail(fmt::format("[{}]: wrote {} methods, read back {}",
                                 join(sizes, ", "), chain.size(), got.size()));
                continue;
            }
            for (size_t i = 0; i < chain.size(); i++) {
                if (got[i].side != chain[i].side)
                    fail(fmt::format("[{}] method {}: side {} became {}",
                                     join(sizes, ", "), i, chain[i].side,
                                     got[i].side));
                if (got[i].method->method != chain[i].method)
                    fail(fmt::format(
                        "[{}] method {}: {} did not survive the round trip",
                        join(sizes, ", "), i, facul_method_name(chain[i])));
                if (chain[i].method == EC_METHOD) {
                    auto const * e = (ecm_plan_t const *) got[i].method->plan;
                    if (e->parameterization != chain[i].parameterization ||
                        e->parameter != chain[i].parameter)
                        fail(fmt::format("[{}] method {}: {}({}) came back"
                                         " with parameter {}",
                                         join(sizes, ", "), i,
                                         facul_method_name(chain[i]),
                                         chain[i].parameter, e->parameter));
                }
                if (got[i].stop_unsplit != chain[i].stop_unsplit)
                    fail(fmt::format("[{}] method {}: the STOP marker did not"
                                     " survive the round trip",
                                     join(sizes, ", "), i));
            }
        }
    } catch (std::exception const & e) {
        fail(fmt::format("the parser rejected what we wrote: {}\n{}", e.what(),
                         text));
    }

    fclose(f);
    return rc;
}
