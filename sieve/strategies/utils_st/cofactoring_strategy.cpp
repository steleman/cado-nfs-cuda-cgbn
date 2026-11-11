#include "cado.h" // IWYU pragma: keep

#include <cctype>

#include <istream>
#include <string>
#include <limits>
#include <ostream>
#include <utility>

#include "fmt/base.h"

#include "cofactoring_strategy.hpp"
#include "utils_cxx.hpp"

std::ostream & operator<<(std::ostream & os, cofactoring_strategy const & s)
{
    for (auto const & step: s.steps) {
        auto const & p = step.method.params;
        os << cofactoring_method_name(p) << ' ' << p.B1 << ' ' << p.B2 << ' '
           << step.side << '\n';
    }
    os << fmt::format("Probability: {:.10f}\n", s.proba);
    os << fmt::format("Time: {:f}\n", s.time);
    return os;
}

std::istream & operator>>(std::istream & is, cofactoring_strategy & s)
{
    cofactoring_strategy r;

    /* One line per method, then the two that close the strategy. The
     * first token tells them apart: method names cannot be
     * "Probability:", and reading it here is what stops the loop --
     * peeking at the first letter would not, since PM1 and PP1-27 start
     * with the same one. */
    for (;;) {
        std::string name;
        if (!(is >> std::ws >> name))
            return is;
        if (name == "Probability:")
            break;

        unsigned long B1, B2;
        int side;
        if (!(is >> B1 >> B2 >> side))
            return is;
        facul_method::parameters params;
        if (!cofactoring_method_by_name(name, params)) {
            is.setstate(std::ios_base::failbit);
            return is;
        }
        r.add(cofactoring_method(cofactoring_method::make_parameters(
                      params.method, params.parameterization, B1, B2)),
              side);
    }

    if (!(is >> r.proba))
        return is;
    if (!(is >> std::ws >> expect("Time:") >> r.time))
        return is;

    s = std::move(r);
    return is;
}

std::ostream & operator<<(std::ostream & os,
                          cofactoring_strategy_list const & t)
{
    for (auto const & s: t)
        os << s;
    return os;
}

std::istream & operator>>(std::istream & is, cofactoring_strategy_list & t)
{
    for (;;) {
        is >> std::ws;
        if (is.eof())
            break;
        if (!is.good())
            return is;
        if (is.peek() == '#') {
            is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        cofactoring_strategy s;
        if (!(is >> s))
            return is;
        t.push_back(std::move(s));
    }
    return is;
}
