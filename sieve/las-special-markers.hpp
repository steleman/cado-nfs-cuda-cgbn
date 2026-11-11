#ifndef CADO_LAS_SPECIAL_MARKERS_HPP
#define CADO_LAS_SPECIAL_MARKERS_HPP

/* las prints these lines among the relations, and other programs read
 * them back: dupsup and fake_rels through the parse() functions below,
 * and a few scripts with regexes, which must follow this list:
 *
 *   # Now sieving <special-q>
 *   # Sieving <special-q>; <q-lattice>; I=<I>; J=<J>;
 *   # Sublattice (i, j) == (<i0>, <j0>) mod <m>
 *   # Time for <special-q>: <timings>
 *   # <n> relations for <special-q>
 *
 * <special-q> is "side-<s> q=<q>; rho=<r>", where a composite q is
 * written <q>=<p1>*<p2>*... (operator<< of special_q). <q-lattice> is
 * "a0=<a0>; b0=<b0>; a1=<a1>; b1=<b1>" for las (qlattice_basis), and
 * "CRT data = [(<p1>, <c1>), ...]; r0 = <r0>" for siqs
 * (siqs_special_q_data); both have an operator>>. Older las output had
 * "r=" instead of "rho=" in the "Time for" lines, which parse() also
 * takes.
 */

#include <cstdint>
#include <cstdio>

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "fmt/base.h"
#include "fmt/format.h"

#include "las-sublat.hpp"
#include "special-q.hpp"
#include "utils_cxx.hpp"

namespace cado::special_markers {

namespace details {
    inline std::optional<special_q> read_special_q(std::string_view s)
    {
        std::istringstream is { std::string(s) };
        special_q Q;
        if (!(is >> Q))
            return {};
        return Q;
    }
} // namespace details

/* "# Now sieving <special-q>" */
struct begin_special_q {
    static constexpr std::string_view prefix = "# Now sieving ";
    special_q Q;

    static std::optional<begin_special_q> parse(std::string_view line)
    {
        if (!line.starts_with(prefix))
            return {};
        auto Q = details::read_special_q(line.substr(prefix.size()));
        if (!Q)
            return {};
        return begin_special_q { std::move(*Q) };
    }
};

/* "# Sieving <special-q>; <q-lattice>; I=<I>; J=<J>;<extra>"
 * D is the q-lattice data, qlattice_basis or siqs_special_q_data, which
 * prints the special-q first, and has its operator>>. extra is what las
 * appends within a descent. */
template <typename D> struct sieving {
    static constexpr std::string_view prefix = "# Sieving ";
    D Q;
    unsigned int I = 0;
    unsigned int J = 0;
    std::string extra {};

    /* what follows the prefix */
    std::string body() const
    {
        return fmt::format("{}; I={}; J={};{}", Q, I, J, extra);
    }

    static std::optional<sieving> parse(std::string_view line)
    {
        if (!line.starts_with(prefix))
            return {};
        std::istringstream is { std::string(line.substr(prefix.size())) };
        sieving m;
        if (!(is >> m.Q >> expect("; I=") >> m.I >> expect("; J=") >> m.J
                    >> expect(";")))
            return {};
        std::getline(is, m.extra);
        return m;
    }
};

/* "# Sublattice (i, j) == (<i0>, <j0>) mod <m>" */
struct begin_sublattice {
    static constexpr std::string_view prefix = "# Sublattice ";
    sublat_runtime_t sl;

    static std::optional<begin_sublattice> parse(std::string_view line)
    {
        if (!line.starts_with(prefix))
            return {};
        std::string const s(line.substr(prefix.size()));
        unsigned int i0, j0, m;
        int n = 0;
        if (sscanf(s.c_str(), "(i, j) == (%u, %u) mod %u%n", &i0, &j0, &m, &n)
                != 3 || size_t(n) != s.size())
            return {};
        return begin_sublattice { sublat_runtime_t(m, i0, j0) };
    }
};

/* "# Time for <special-q>:" -- las prints the timings after it */
struct end_special_q {
    static constexpr std::string_view prefix = "# Time for ";
    special_q Q;

    static std::optional<end_special_q> parse(std::string_view line)
    {
        if (!line.starts_with(prefix))
            return {};
        /* the colon would stick to rho, which then does not parse */
        line.remove_prefix(prefix.size());
        auto Q = details::read_special_q(line.substr(0, line.find(':')));
        if (!Q)
            return {};
        return end_special_q { std::move(*Q) };
    }
};

/* "# <n> relations for <special-q>" -- the relations of the batch, which
 * come after it */
struct batch_relations {
    static constexpr std::string_view prefix = "# ";
    static constexpr std::string_view infix = " relations for ";
    size_t n;
    special_q Q;

    static std::optional<batch_relations> parse(std::string_view line)
    {
        if (!line.starts_with(prefix))
            return {};
        line.remove_prefix(prefix.size());
        size_t const k = line.find(infix);
        if (k == 0 || k == std::string_view::npos)
            return {};
        size_t n = 0;
        for (char const c: line.substr(0, k)) {
            if (c < '0' || c > '9')
                return {};
            n = 10 * n + (c - '0');
        }
        auto Q = details::read_special_q(line.substr(k + infix.size()));
        if (!Q)
            return {};
        return batch_relations { n, std::move(*Q) };
    }
};

} // namespace cado::special_markers

namespace fmt {
template <> struct formatter<cado::special_markers::begin_special_q>
    : formatter<string_view> {
    auto format(cado::special_markers::begin_special_q const & m,
                format_context & ctx) const
    {
        return fmt::format_to(ctx.out(), "{}{}", m.prefix, m.Q);
    }
};
template <typename D> struct formatter<cado::special_markers::sieving<D>>
    : formatter<string_view> {
    auto format(cado::special_markers::sieving<D> const & m,
                format_context & ctx) const
    {
        return fmt::format_to(ctx.out(), "{}{}", m.prefix, m.body());
    }
};
template <> struct formatter<cado::special_markers::begin_sublattice>
    : formatter<string_view> {
    auto format(cado::special_markers::begin_sublattice const & m,
                format_context & ctx) const
    {
        return fmt::format_to(ctx.out(), "{}{}", m.prefix, m.sl);
    }
};
template <> struct formatter<cado::special_markers::end_special_q>
    : formatter<string_view> {
    auto format(cado::special_markers::end_special_q const & m,
                format_context & ctx) const
    {
        return fmt::format_to(ctx.out(), "{}{}:", m.prefix, m.Q);
    }
};
template <> struct formatter<cado::special_markers::batch_relations>
    : formatter<string_view> {
    auto format(cado::special_markers::batch_relations const & m,
                format_context & ctx) const
    {
        return fmt::format_to(ctx.out(), "{}{}{}{}", m.prefix, m.n, m.infix,
                              m.Q);
    }
};
} // namespace fmt

#endif /* CADO_LAS_SPECIAL_MARKERS_HPP */
