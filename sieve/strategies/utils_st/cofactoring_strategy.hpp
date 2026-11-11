#ifndef CADO_COFACTORING_STRATEGY_HPP
#define CADO_COFACTORING_STRATEGY_HPP

#include <cstddef>

#include <istream>
#include <ostream>
#include <vector>

#include "fmt/base.h"
#include "fmt/ostream.h"

#include "facul_method.hpp"

#include "cofactoring_method.hpp"

/* A chain of methods to run on a pair of cofactors, with the probability
 * that it finds a non-trivial factor and the average time it takes.
 *
 * This is the thing the whole directory exists to produce: what choose_cofactoring_strategies
 * writes, and what las reads back with -file-cofact.
 */
struct cofactoring_strategy {
    /* One step of the chain: a method, and the side it applies to.
     *
     * That pair is facul_method::parameters_with_side, which is what las
     * ends up with -- see parameters() below. What a step carries on top
     * of it is what the benchmark measured about the method, which the
     * generator needs to predict the probability and the cost of the
     * chain, and which the file las reads does not hold. */
    struct step {
        cofactoring_method method;
        /* Strategies are built one side at a time, and everything stays
         * on side 0 until two of them are concatenated. */
        int side = 0;

        facul_method::parameters_with_side parameters() const
        {
            return {side, method.params};
        }
    };

    std::vector<step> steps;
    double proba = 0;
    double time = 0;

    void add(cofactoring_method const & m, int side = 0)
    {
        steps.push_back({m, side});
    }

    size_t size() const { return steps.size(); }
    bool empty() const { return steps.empty(); }

    /* The chain, in the form las ends up holding: this is what
     * facul_strategies::strategy_file maps each pair of cofactor sizes
     * to. Everything a step carries beyond it -- what the benchmark
     * measured about the method -- is scaffolding for the search, and is
     * already gone by the time a strategy has been written to a file.
     * So are the zero methods, which las has no name for: a strategy
     * made only of them is an empty chain, which las skips. */
    std::vector<facul_method::parameters_with_side> as_chain() const
    {
        std::vector<facul_method::parameters_with_side> v;
        v.reserve(steps.size());
        for (auto const & s: steps)
            if (!s.method.is_zero())
                v.push_back(s.parameters());
        return v;
    }
};

using cofactoring_strategy_list = std::vector<cofactoring_strategy>;

std::istream & operator>>(std::istream & is, cofactoring_strategy &);
std::ostream & operator<<(std::ostream & os, cofactoring_strategy const &);

std::istream & operator>>(std::istream & is, cofactoring_strategy_list &);
std::ostream & operator<<(std::ostream & os, cofactoring_strategy_list const &);

namespace fmt
{
template <> struct formatter<cofactoring_strategy> : ostream_formatter {
};
template <> struct formatter<cofactoring_strategy_list> : ostream_formatter {
};
} // namespace fmt

#endif /* CADO_COFACTORING_STRATEGY_HPP */
