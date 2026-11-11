#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "ecm/facul_method.hpp"
#include "ecm/facul_strategies.hpp"
#include "macros.h"
#include "utils_cxx.hpp"

/* The strategy file syntax: what fragments, slices, ranges and side
 * switches mean (against the same file spelled out), that what the
 * writer writes reads back, and the errors. */

using chain_t = std::vector<facul_method::parameters_with_side>;
using sfile = facul_strategies::strategy_file;

static std::vector<unsigned int> const mfb { 40, 40 };

static sfile read_text(std::string const & text)
{
    // coverity[secure_temp]
    FILE * f = tmpfile();
    DIE_ERRNO_DIAG(f == nullptr, "tmpfile(%s)", "");
    fputs(text.c_str(), f);
    fflush(f);
    sfile res;
    try {
        res = facul_strategies::read_file(mfb, f);
    } catch (...) {
        fclose(f);
        throw;
    }
    fclose(f);
    return res;
}

static std::string show(chain_t const & c)
{
    std::ostringstream os;
    for (auto const & m: c)
        os << m.side << ":" << m << " ";
    return os.str();
}

static bool same(sfile const & a, sfile const & b)
{
    if (a.size() != b.size())
        return false;
    for (auto const & [k, c]: a) {
        auto it = b.find(k);
        if (it == b.end() || show(c) != show(it->second))
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

    std::string const text = R"(# comments anywhere; newlines are not significant
p = { PM1 315 2205 PP1-27 525 3255 }
e = {
    ECM-TM12(2) 105 3255
    ECM-TM12(1) 315 5355 STOP
}
q = { 1: PM1 315 2205 }         # a side inside a fragment stays inside
10-11 20: { 1: p e 0: p MPQS }
12 20: { 0: p[1:] 1: e[:1]
         1: PM1 100 1000 }
13 20-21: { 1: p }
13 21: { }                      # rejects a cell set above
14 20:{q PM1 100 1000}
15 20: { ECM-B12 315 5355 ECM-B12 315 5355 }
)";
    std::string const spelled = R"(
10 20: {
    1: PM1 315 2205
    1: PP1-27 525 3255
    1: ECM-TM12(2) 105 3255
    1: ECM-TM12(1) 315 5355 STOP
    0: PM1 315 2205
    0: PP1-27 525 3255
    0: MPQS
}
11 20: { 1: PM1 315 2205 1: PP1-27 525 3255 1: ECM-TM12(2) 105 3255
         1: ECM-TM12(1) 315 5355 STOP 0: PM1 315 2205 0: PP1-27 525 3255 0: MPQS }
12 20: { 0: PP1-27 525 3255 1: ECM-TM12(2) 105 3255 1: PM1 100 1000 }
13 20: { 1: PM1 315 2205 1: PP1-27 525 3255 }
14 20: { 1: PM1 315 2205 0: PM1 100 1000 }
15 20: { 0: ECM-B12 315 5355 0: ECM-B12 315 5355 }
)";

    try {
        sfile const S = read_text(text);
        sfile const F = read_text(spelled);
        if (!same(S, F)) {
            fail("the file and its spelled-out version do not read the same");
            for (auto const & [k, c]: S)
                fmt::print(stderr, "  new {}: {}\n", join(k, " "), show(c));
            for (auto const & [k, c]: F)
                fmt::print(stderr, "  spelled out {}: {}\n", join(k, " "), show(c));
        }
        /* the bare names took the default sequence: 11, then the next */
        auto const & b = S.at({15, 20});
        if (b[0].parameter != 11 || b[1].parameter == 11)
            fail(fmt::format("bare ECM-B12 got parameters {} and {}", b[0].parameter, b[1].parameter));
        /* and what the writer writes reads back */
        std::ostringstream os;
        facul_strategies::write_file(os, S);
        if (!same(read_text(os.str()), S))
            fail("write_file() output does not read back:\n" + os.str());
    } catch (std::exception const & e) {
        fail(fmt::format("unexpected parse error: {}", e.what()));
    }

    for (auto const * bad: {
            "10 20: { FOO 1 2 }\n",            /* neither method nor fragment */
            "10 20: { 1: undefined }\n",
            "p = { PM1 315 2205 }\n10 20: { p[2:1] }\n",
            "p = { PM1 315 2205 }\n10 20: { p[0:2] }\n",
            "p = { PM1 315 2205 }\n10: { p }\n",        /* one size for two sides */
            "p = { PM1 315 2205 }\np = { PM1 315 2205 }\n",
            "PM1 = { PM1 315 2205 }\n",
            "10 20: { STOP PM1 315 2205 }\n",
            "10 20: { PM1 315 }\n",
            "10 20: { 2: PM1 315 2205 }\n",    /* no side 2 */
            "r = { r }\n",
            "10 20: PM1 315 2205\n",           /* no braces */
            "10 20: { PM1 315 2205\n",         /* no closing brace */
            "10 20: { PM1 315 2205 } }\n",
            }) {
        try {
            read_text(bad);
            fail(fmt::format("accepted: {}", bad));
        } catch (std::exception const &) {
        }
    }
    return rc;
}
