#include "cado.h" // IWYU pragma: keep

#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fmt/format.h"
#include "fmt/ranges.h"

#include "facul_strategies.hpp"
#include "facul_ecm.h"
#include "facul_method.hpp"
#include "pm1.hpp"
#include "pp1.hpp"
#include "macros.h"
#include "verbose.hpp"

//#define USE_LEGACY_DEFAULT_STRATEGY 1

/*{{{*/
static int nb_curves90 (const unsigned int lpb);
#if 0
static int nb_curves95 (const unsigned int lpb);
static int nb_curves99 (const unsigned int lpb);
#endif

/* don't use nb_curves90, nb_curves95 and nb_curves99, only use nb_curves */
int nb_curves (const unsigned int lpb, const unsigned int mfb)
{
  /* for up to 2 large primes, we use a very large number of curves (say 100),
     since the more curves we need, the more likely the cofactor is smooth. */
  if (mfb <= 2 * lpb)
    return 100;
  return nb_curves90 (lpb);
}

static int
nb_curves_with_fbb (const unsigned long fbb,
		    const unsigned int lpb, const unsigned int mfb)
{
  /* if 2^mfb < fbb^2, we can have only one large prime */
  return (ldexp (1.0, mfb) < (double) fbb * (double) fbb)
    ? 0 : nb_curves (lpb, mfb);
}

static int
nb_curves90 (const unsigned int lpb)
{
  /* The following table, computed with do_table(10,33,ntries=10000) from the
     facul.sage file, ensures a probability of at least about 90% to find a
     factor below 2^lpb with n = T[lpb]. */
  int const T[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 0-9 */
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 10-19 */
	     /* lpb=20 */ 1, /* 0:0.878100, 1:0.969600 */
	     /* lpb=21 */ 1, /* 1:0.940400 */
	     /* lpb=22 */ 1, /* 1:0.907400 */
	     /* lpb=23 */ 2, /* 1:0.859100, 2:0.903900 */
	     /* lpb=24 */ 4, /* 3:0.897000, 4:0.929200 */
	     /* lpb=25 */ 5, /* 4:0.884100, 5:0.916400 */
	     /* lpb=26 */ 6, /* 5:0.868600, 6:0.904200 */
	     /* lpb=27 */ 8, /* 7:0.873500, 8:0.901700 */
             /* lpb=28 */ 11, /* 10:0.896600, 11:0.918000 */
	     /* lpb=29 */ 13, /* 12:0.881700, 13:0.905600 */
	     /* lpb=30 */ 16, /* 15:0.889400, 16:0.909400 */
	     /* lpb=31 */ 18, /* 17:0.883500, 18:0.901300 */
	     /* lpb=32 */ 21, /* 20:0.884500, 21:0.903700 */
	     /* lpb=33 */ 25, /* 24:0.892800, 25:0.910000 */
        /* The extra ones below are computed with do_table(10,64,ntries=200) */
	     /* lpb=34 */ 24, /* 23:0.895000, 24:0.905000 */
	     /* lpb=35 */ 29, /* 28:0.895000, 29:0.905000 */
	     /* lpb=36 */ 30, /* 29:0.890000, 30:0.905000 */
	     /* lpb=37 */ 35, /* 34:0.890000, 35:0.900000 */
	     /* lpb=38 */ 37, /* 36:0.890000, 37:0.900000 */
	     /* lpb=39 */ 42, /* 41:0.890000, 42:0.900000 */
	     /* lpb=40 */ 45, /* 44:0.890000, 45:0.900000 */
	     /* lpb=41 */ 50, /* 49:0.885000, 50:0.905000 */
	     /* lpb=42 */ 56, /* 55:0.875000, 56:0.900000 */
	     /* lpb=43 */ 60, /* 59:0.895000, 60:0.900000 */
	     /* lpb=44 */ 68, /* 67:0.890000, 68:0.910000 */
	     /* lpb=45 */ 73, /* 72:0.895000, 73:0.905000 */
	     /* lpb=46 */ 82, /* 81:0.895000, 82:0.900000 */
	     /* lpb=47 */ 90, /* 89:0.895000, 90:0.900000 */
	     /* lpb=48 */ 93, /* 92:0.895000, 93:0.900000 */
	     /* lpb=49 */ 93, /* 93:0.900000 */
	     /* lpb=50 */ 111, /* 110:0.885000, 111:0.900000 */
	     /* lpb=51 */ 117, /* 116:0.890000, 117:0.905000 */
	     /* those below were computed with do_table(10,64,ntries=100) */
	     /* lpb=52 */ 130, /* 129:0.890000, 130:0.900000 */
	     /* lpb=53 */ 137, /* 136:0.890000, 137:0.900000 */
	     /* lpb=54 */ 142, /* 141:0.890000, 142:0.900000 */
	     /* lpb=55 */ 152, /* 151:0.890000, 152:0.920000 */
	     /* lpb=56 */ 167, /* 166:0.890000, 167:0.900000 */
	     /* lpb=57 */ 167, /* 167:0.910000 */
	     /* lpb=58 */ 186, /* 185:0.890000, 186:0.900000 */
	     /* lpb=59 */ 202, /* 201:0.880000, 202:0.900000 */
	     /* lpb=60 */ 227, /* 226:0.890000, 227:0.910000 */
	     /* lpb=61 */ 244, /* 243:0.890000, 244:0.910000 */
	     /* lpb=62 */ 244, /* 244:0.920000 */
	     /* lpb=63 */ 248, /* 247:0.880000, 248:0.900000 */
	     /* lpb=64 */ 294, /* 293:0.890000, 294:0.900000 */
  };
  const unsigned int nT = sizeof(T)/sizeof(int) - 1;
  return (lpb <= nT) ? T[lpb] : T[nT];
}

#if 0
static int
nb_curves95 (const unsigned int lpb)
{
    /* same, but with target probability 95% */
    /* do_table(10,64,ntries=500,target_prob=0.95)
     */
  int T[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 0-9 */
	     /* lpb=10 */ 0, /* 0:1.000000 */
	     /* lpb=11 */ 0, /* 0:1.000000 */
	     /* lpb=12 */ 0, /* 0:1.000000 */
	     /* lpb=13 */ 0, /* 0:1.000000 */
	     /* lpb=14 */ 0, /* 0:1.000000 */
	     /* lpb=15 */ 0, /* 0:0.998000 */
	     /* lpb=16 */ 0, /* 0:0.986000 */
	     /* lpb=17 */ 0, /* 0:0.972000 */
	     /* lpb=18 */ 0, /* 0:0.964000 */
	     /* lpb=19 */ 1, /* 0:0.926000, 1:0.986000 */
	     /* lpb=20 */ 1, /* 1:0.986000 */
	     /* lpb=21 */ 2, /* 1:0.946000, 2:0.976000 */
	     /* lpb=22 */ 3, /* 2:0.940000, 3:0.966000 */
	     /* lpb=23 */ 3, /* 3:0.952000 */
	     /* lpb=24 */ 5, /* 4:0.926000, 5:0.962000 */
	     /* lpb=25 */ 7, /* 6:0.934000, 7:0.956000 */
	     /* lpb=26 */ 8, /* 7:0.934000, 8:0.956000 */
	     /* lpb=27 */ 10, /* 9:0.936000, 10:0.956000 */
	     /* lpb=28 */ 13, /* 12:0.938000, 13:0.954000 */
	     /* lpb=29 */ 16, /* 15:0.946000, 16:0.956000 */
	     /* lpb=30 */ 18, /* 17:0.936000, 18:0.950000 */
	     /* lpb=31 */ 22, /* 21:0.934000, 22:0.958000 */
	     /* lpb=32 */ 26, /* 25:0.940000, 26:0.950000 */
	     /* lpb=33 */ 29, /* 28:0.942000, 29:0.952000 */
	     /* lpb=34 */ 33, /* 32:0.948000, 33:0.956000 */
	     /* lpb=35 */ 37, /* 36:0.938000, 37:0.952000 */
	     /* lpb=36 */ 42, /* 41:0.946000, 42:0.952000 */
	     /* lpb=37 */ 45, /* 44:0.946000, 45:0.958000 */
  };
  const unsigned int nT = sizeof(T)/sizeof(int) - 1;
  return (lpb <= nT) ? T[lpb] : T[nT];
}

static int
nb_curves99 (const unsigned int lpb)
{
    /* same, but with target probability 99% */
    /* do_table(10,64,ntries=100,target_prob=0.99)
     */
  int T[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 0-9 */
	     /* lpb=10 */ 0, /* 0:1.000000 */
	     /* lpb=11 */ 0, /* 0:1.000000 */
	     /* lpb=12 */ 0, /* 0:1.000000 */
	     /* lpb=13 */ 0, /* 0:1.000000 */
	     /* lpb=14 */ 0, /* 0:1.000000 */
	     /* lpb=15 */ 0, /* 0:1.000000 */
	     /* lpb=16 */ 0, /* 0:1.000000 */
	     /* lpb=17 */ 1, /* 0:0.980000, 1:1.000000 */
	     /* lpb=18 */ 1, /* 1:0.990000 */
	     /* lpb=19 */ 2, /* 1:0.980000, 2:1.000000 */
	     /* lpb=20 */ 2, /* 2:0.990000 */
	     /* lpb=21 */ 2, /* 2:0.990000 */
	     /* lpb=22 */ 4, /* 3:0.980000, 4:1.000000 */
	     /* lpb=23 */ 4, /* 4:0.990000 */
	     /* lpb=24 */ 5, /* 4:0.950000, 5:0.990000 */
	     /* lpb=25 */ 8, /* 7:0.980000, 8:0.990000 */
	     /* lpb=26 */ 12, /* 11:0.980000, 12:1.000000 */
	     /* lpb=27 */ 14, /* 13:0.980000, 14:1.000000 */
	     /* lpb=28 */ 14, /* 14:1.000000 */
	     /* lpb=29 */ 14, /* 14:0.990000 */
	     /* lpb=30 */ 17, /* 16:0.970000, 17:0.990000 */
	     /* lpb=31 */ 25, /* 24:0.980000, 25:1.000000 */
	     /* lpb=32 */ 25, /* 25:0.990000 */
	     /* lpb=33 */ 25, /* 25:0.990000 */
	     /* lpb=34 */ 32, /* 31:0.980000, 32:0.990000 */
	     /* lpb=35 */ 36, /* 35:0.980000, 36:0.990000 */
	     /* lpb=36 */ 39, /* 38:0.980000, 39:0.990000 */
	     /* lpb=37 */ 43, /* 42:0.980000, 43:0.990000 */
	     /* lpb=38 */ 49, /* 48:0.980000, 49:0.990000 */
	     /* lpb=39 */ 52, /* 51:0.970000, 52:0.990000 */
	     /* lpb=40 */ 56, /* 55:0.980000, 56:0.990000 */
	     /* lpb=41 */ 60, /* 59:0.980000, 60:0.990000 */
  };
  const unsigned int nT = sizeof(T)/sizeof(int) - 1;
  return (lpb <= nT) ? T[lpb] : T[nT];
}
#endif
/*}}}*/

/* TODO: move elsewhere. */
const char * parameterization_name(ec_parameterization_t p)
{
    switch(p) {
        case BRENT12:     return "ECM-B12";
        case MONTY12:     return "ECM-M12";
        case MONTY16:     return "ECM-M16";
        case MONTYTWED12: return "ECM-TM12";
        case MONTYTWED16: return "ECM-TM16";
    }
    ASSERT_ALWAYS(0);
    return nullptr;
}

std::string facul_method_name(facul_method::parameters const & p)
{
    switch (p.method) {
        case PM1_METHOD:    return "PM1";
        case PP1_27_METHOD: return "PP1-27";
        case PP1_65_METHOD: return "PP1-65";
        case EC_METHOD:     return parameterization_name(p.parameterization);
        case MPQS_METHOD:   return "MPQS";
        default: break;
    }
    ASSERT_ALWAYS(0);
    return {};
}

std::vector<facul_method::parameters> facul_method_families()
{
    std::vector<facul_method::parameters> res;
    for (auto const m: {PM1_METHOD, PP1_27_METHOD, PP1_65_METHOD})
        res.emplace_back(m, 0UL, 0UL);
    /* every parameterization the enum names, less the ones nothing can
     * build a curve for */
    for (auto const p: {BRENT12, MONTY12, MONTY16, MONTYTWED12, MONTYTWED16})
        if (ec_parameterization_is_implemented(p))
            res.emplace_back(EC_METHOD, 0UL, 0UL, p, 0UL, 1);
    return res;
}

bool facul_method_by_name(std::string const & name,
                          facul_method::parameters & res)
{
    /* MPQS takes no bounds (B1 and B2 are ignored). It is not in
     * facul_method_families(), which the strategy builders sweep. */
    if (name == "MPQS") {
        res.method = MPQS_METHOD;
        return true;
    }
    for (auto const & f: facul_method_families()) {
        if (facul_method_name(f) != name)
            continue;
        res.method = f.method;
        res.parameterization = f.parameterization;
        return true;
    }
    return false;
}

std::string facul_method_family_names()
{
    std::string res;
    for (auto const & f: facul_method_families()) {
        if (!res.empty())
            res += ", ";
        res += facul_method_name(f);
    }
    return res;
}

std::ostream & operator<<(std::ostream & os,
                          facul_method::parameters_with_side const & m)
{
    /* parameter 0 means "the next one of the default sequence", which
     * is what a bare name gives; any other one is spelled out */
    std::string name = facul_method_name(m);
    if (m.method == EC_METHOD && m.parameter != 0)
        name += fmt::format("({})", m.parameter);
    if (m.method == MPQS_METHOD)
        os << name;
    else
        os << fmt::format("{} {} {}", name, m.B1, m.B2);
    if (m.stop_unsplit)
        os << " STOP";
    return os;
}

namespace {
/* one side of a chain, as the tokens of its methods */
using side_chain = std::vector<std::string>;

std::string range_text(unsigned int a, unsigned int b)
{
    return a == b ? fmt::format("{}", a) : fmt::format("{}-{}", a, b);
}
}

/* The chains of all cells, in the syntax that read_file() reads: the
 * side chains that are not a prefix of another one, if used more than
 * once, become fragments (d, d1, ...), each cell is a sequence of slices
 * of them, one side at a time, and cells with the same chain are merged
 * into ranges. */
void facul_strategies::write_file(std::ostream & os, strategy_file const & S)
{
    std::map<std::vector<unsigned int>, std::vector<side_chain>> sides;
    for (auto const & [k, chain]: S) {
        auto & v = sides[k];
        v.resize(k.size());
        for (auto const & m: chain) {
            std::ostringstream t;
            t << m;
            v[m.side].push_back(t.str());
        }
    }
    std::vector<side_chain> distinct;
    for (auto const & [k, v]: sides)
        for (auto const & c: v)
            if (!c.empty())
                distinct.push_back(c);
    std::ranges::sort(distinct, [](auto const & a, auto const & b) {
            return a.size() > b.size() || (a.size() == b.size() && a < b); });
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    std::vector<side_chain> bases;
    auto is_prefix = [](side_chain const & c, side_chain const & b) {
        return c.size() <= b.size() && std::equal(c.begin(), c.end(), b.begin());
    };
    auto base_of = [&](side_chain const & c) {
        for (size_t i = 0 ; i < bases.size() ; i++)
            if (is_prefix(c, bases[i]))
                return i;
        ASSERT_ALWAYS(0);
        return size_t(0);
    };
    for (auto const & c: distinct)
        if (std::ranges::none_of(bases, [&](auto const & b) { return is_prefix(c, b); }))
            bases.push_back(c);
    std::vector<unsigned int> uses(bases.size(), 0);
    for (auto const & [k, v]: sides)
        for (auto const & c: v)
            if (!c.empty())
                uses[base_of(c)]++;
    std::vector<std::string> names(bases.size());
    for (size_t i = 0, n = 0 ; i < bases.size() ; i++)
        if (uses[i] > 1)
            names[i] = n++ ? fmt::format("d{}", n - 1) : "d";

    /* each cell: runs of one side, each a slice of that side's chain;
     * one item per run, so that a long chain can go one per line */
    std::map<std::vector<std::string>, std::vector<std::vector<unsigned int>>> by_text;
    for (auto const & [k, chain]: S) {
        auto const & v = sides.at(k);
        std::vector<size_t> pos(k.size(), 0);
        std::vector<std::string> items;
        for (size_t i = 0 ; i < chain.size() ; ) {
            int const s = chain[i].side;
            size_t j = i;
            while (j < chain.size() && chain[j].side == s)
                j++;
            size_t const a = pos[s], e = a + (j - i);
            auto const & c = v[s];
            size_t const b = base_of(c);
            std::string const sw = fmt::format("{}: ", s);
            if (names[b].empty()) {
                for (size_t t = a ; t < e ; t++)
                    items.push_back((t == a ? sw : "") + c[t]);
            } else if (a == 0 && e == bases[b].size()) {
                items.push_back(sw + names[b]);
            } else {
                items.push_back(sw + fmt::format("{}[{}:{}]", names[b], a ? fmt::format("{}", a) : "", e));
            }
            pos[s] = e;
            i = j;
        }
        by_text[items].push_back(k);
    }

    /* cells with the same chain: runs along the last size, then the
     * runs that repeat along the first one, for two sides */
    struct line {
        std::vector<std::pair<unsigned int, unsigned int>> ranges;
        std::vector<std::string> items;
    };
    std::vector<line> lines;
    for (auto const & [text, keys]: by_text) {
        std::map<std::vector<unsigned int>, std::vector<std::pair<unsigned int, unsigned int>>> runs;
        for (size_t i = 0 ; i < keys.size() ; ) {
            std::vector<unsigned int> head(keys[i].begin(), keys[i].end() - 1);
            unsigned int const a = keys[i].back();
            unsigned int b = a;
            size_t j = i + 1;
            for ( ; j < keys.size() ; j++) {
                std::vector<unsigned int> const h(keys[j].begin(), keys[j].end() - 1);
                if (h != head || keys[j].back() != b + 1)
                    break;
                b++;
            }
            runs[head].emplace_back(a, b);
            i = j;
        }
        for (auto it = runs.begin() ; it != runs.end() ; ) {
            auto jt = std::next(it);
            unsigned int last = it->first.empty() ? 0 : it->first[0];
            if (it->first.size() == 1) {
                while (jt != runs.end() && jt->first[0] == last + 1 && jt->second == it->second) {
                    last++;
                    ++jt;
                }
            }
            for (auto const & r: it->second) {
                line L;
                for (size_t i = 0 ; i < it->first.size() ; i++)
                    L.ranges.emplace_back(it->first[i], i == 0 ? last : it->first[i]);
                L.ranges.push_back(r);
                L.items = text;
                lines.push_back(L);
            }
            it = jt;
        }
    }
    std::ranges::sort(lines, {}, &line::ranges);

    for (size_t i = 0 ; i < bases.size() ; i++)
        if (!names[i].empty())
            os << fmt::format("{} = {{\n    {}\n}}\n", names[i], fmt::join(bases[i], "\n    "));
    for (auto const & L: lines) {
        std::vector<std::string> r;
        for (auto const & [a, b]: L.ranges)
            r.push_back(range_text(a, b));
        std::string const one = fmt::format("{}: {{ {} }}", fmt::join(r, " "), fmt::join(L.items, " "));
        if (one.size() <= 80)
            os << one << "\n";
        else
            os << fmt::format("{}: {{\n    {}\n}}\n", fmt::join(r, " "), fmt::join(L.items, "\n    "));
    }
}

struct strategy_file_error : public std::runtime_error {
    explicit strategy_file_error(std::string const & s) : std::runtime_error(s) {}
};

class parameter_sequence_tracker {/*{{{*/
    std::map<unsigned int, std::vector<std::pair<bool, unsigned long>>> seq;
    public:
    parameter_sequence_tracker() {
        decltype(seq)::mapped_type const Z;
        seq[BRENT12] = Z;
        seq[MONTY12] = Z;
        seq[MONTY16] = Z;
        seq[MONTYTWED12] = Z;
        /* we have no parameter_from_sequence for MONTY16
         * anyway. FIXME */
        seq[MONTYTWED16] = Z;
    }

    /* where the sequence of this parameterization is, on this side */
    std::pair<bool, unsigned long> & at(ec_parameterization_t para, int side)
    {
        auto & v = seq[para];
        if (v.size() <= size_t(side))
            v.resize(side + 1, { true, 0 });
        return v[side];
    }

    unsigned long next_in_sequence(int side, ec_parameterization_t para)
    {
        /* If the sequence has been broken, we can no longer give a
         * "next" value. Of course we could improve this by recording the
         * out-of-sequence parameters that have been chosen, and exclude
         * them from the default choices.
         */
        if (!at(para, side).first)
            throw strategy_file_error(fmt::format("cannot use a parameter from the default sequence for parameterization {} after a non-default one was set", parameterization_name(para)));
        return ec_valid_parameter_from_sequence(para, at(para, side).second++);
    }
    void break_sequence(int side, ec_parameterization_t para)
    {
        at(para, side).first = false;
    }

    static void fill_default_parameters(std::vector<facul_method::parameters_with_side> & v) {
        parameter_sequence_tracker tracker;

        for(auto & fm : v) {
            if (fm.method == EC_METHOD) {
                if (fm.parameter == ULONG_MAX) {
                    fm.parameter = tracker.next_in_sequence(fm.side, fm.parameterization);
                } else {
                    tracker.break_sequence(fm.side, fm.parameterization);
                }
                /* it seems that we always have this on */
                fm.extra_primes = 1;
            }
        }
    }
};/*}}}*/

/* The syntax that facul_strategies::write_file() writes; newlines are
 * not significant, and # starts a comment that runs to the end of the
 * line:
 *     name = { CHAIN }          a fragment
 *     R0 R1: { CHAIN }          a cell; R is n or a-b (all cells in between)
 * A CHAIN is a sequence of
 *     NAME B1 B2                a method; NAME as in facul_method_by_name(),
 *                               with an optional curve parameter, ECM-M12(3)
 *     MPQS                      (no bounds)
 *     STOP                      after a method of the current side
 *     s:                        the side of the methods that follow (0 by
 *                               default)
 *     name, name[a:b]           a fragment, or methods a to b-1 of it
 *                               (both optional, as in Python); the side
 *                               switches inside a fragment stay inside it
 * A later cell statement overrides an earlier one; an empty chain
 * rejects the cell. */
static facul_strategies::strategy_file
parse_strategy_file(std::vector<unsigned int> const & mfb, std::string const & text)
{
    using error = strategy_file_error;
    using chain_t = std::vector<facul_method::parameters_with_side>;
    /* the words of the file, with their line numbers; braces and '='
     * are words of their own */
    struct statement {
        int lnum;
        std::vector<std::string> head, chain;
    };
    std::vector<statement> statements;
    {
        std::vector<std::pair<int, std::string>> words;
        std::istringstream is(text);
        std::string line;
        for (int lnum = 1 ; std::getline(is, line) ; lnum++) {
            line = line.substr(0, line.find('#'));
            for (char const c: std::string_view("{}="))
                for (size_t i = 0 ; (i = line.find(c, i)) != std::string::npos ; i += 3)
                    line.replace(i, 1, std::string(" ") + c + " ");
            std::istringstream ls(line);
            for (std::string w ; ls >> w ; )
                words.emplace_back(lnum, w);
        }
        for (size_t i = 0 ; i < words.size() ; ) {
            statement st { words[i].first, {}, {} };
            for ( ; i < words.size() && words[i].second != "{" ; i++) {
                if (words[i].second == "}")
                    throw cado::error("Parse error on line {} of strategies file: '}}' without '{{'", words[i].first);
                st.head.push_back(words[i].second);
            }
            if (i == words.size())
                throw cado::error("Parse error on line {} of strategies file: a chain must be enclosed in {{ }}", st.lnum);
            for (i++ ; i < words.size() && words[i].second != "}" ; i++) {
                if (words[i].second == "{" || words[i].second == "=")
                    throw cado::error("Parse error on line {} of strategies file: missing '}}'", words[i].first);
                st.chain.push_back(words[i].second);
            }
            if (i == words.size())
                throw cado::error("Parse error on line {} of strategies file: missing '}}'", st.lnum);
            i++;
            statements.push_back(std::move(st));
        }
    }
    auto tokens = [](std::string const & t) {
        std::istringstream is(t);
        std::vector<std::string> res;
        for (std::string w ; is >> w ; )
            res.push_back(w);
        return res;
    };
    auto is_name = [](std::string const & t) {
        return !t.empty() && (isalpha((unsigned char) t[0]) || t[0] == '_')
            && std::ranges::all_of(t, [](char c) { return isalnum((unsigned char) c) || c == '_'; });
    };
    auto number = [](std::string const & t) {
        size_t pos;
        unsigned long const v = std::stoul(t, &pos);
        if (pos != t.size())
            throw error(fmt::format("not a number: {}", t));
        return v;
    };

    std::map<std::string, std::vector<std::string>> fragments;
    std::function<chain_t(std::vector<std::string> const &, int, int)> expand;
    expand = [&](std::vector<std::string> const & toks, int side, int depth) {
        if (depth > 32)
            throw error("fragments nested too deep (recursive?)");
        chain_t res;
        for (size_t i = 0 ; i < toks.size() ; ) {
            std::string const & t = toks[i];
            if (t.size() >= 2 && t.back() == ':' && std::all_of(t.begin(), t.end() - 1, [](char c) { return isdigit((unsigned char) c); })) {
                side = int(number(t.substr(0, t.size() - 1)));
                if (side >= int(mfb.size()))
                    throw error(fmt::format("no side {}", side));
                i++;
            } else if (t == "STOP") {
                auto it = std::find_if(res.rbegin(), res.rend(),
                        [&](auto const & m) { return m.side == side; });
                if (it == res.rend())
                    throw error("STOP with no method before it");
                it->stop_unsplit = true;
                i++;
            } else if (auto const lb = t.find('['); is_name(t.substr(0, lb)) && fragments.contains(t.substr(0, lb))) {
                chain_t f = expand(fragments.at(t.substr(0, lb)), side, depth + 1);
                if (lb != std::string::npos) {
                    auto const colon = t.find(':', lb);
                    if (colon == std::string::npos || t.back() != ']')
                        throw error(fmt::format("bad slice {}", t));
                    std::string const sa = t.substr(lb + 1, colon - lb - 1);
                    std::string const se = t.substr(colon + 1, t.size() - colon - 2);
                    size_t const a = sa.empty() ? 0 : number(sa);
                    size_t const e = se.empty() ? f.size() : number(se);
                    if (a > e || e > f.size())
                        throw error(fmt::format("slice {} of a fragment of {} methods", t, f.size()));
                    f = chain_t(f.begin() + a, f.begin() + e);
                }
                res.insert(res.end(), f.begin(), f.end());
                i++;
            } else {
                facul_method::parameters_with_side m;
                m.side = side;
                m.parameter = ULONG_MAX;
                std::string name = t;
                if (auto const lp = t.find('('); lp != std::string::npos) {
                    if (t.back() != ')')
                        throw error(fmt::format("cannot parse {}", t));
                    m.parameter = number(t.substr(lp + 1, t.size() - lp - 2));
                    name = t.substr(0, lp);
                }
                if (!facul_method_by_name(name, m))
                    throw error(fmt::format("{} is neither a method ({}, MPQS) nor a fragment",
                                t, facul_method_family_names()));
                i++;
                if (m.method != MPQS_METHOD) {
                    if (i + 2 > toks.size())
                        throw error(fmt::format("{} needs B1 and B2", t));
                    m.B1 = number(toks[i]);
                    m.B2 = number(toks[i + 1]);
                    i += 2;
                }
                res.push_back(m);
            }
        }
        return res;
    };

    facul_strategies::strategy_file parsed;
    for (auto const & [lnum, head0, chain_words]: statements) {
        try {
            if (std::ranges::find(head0, "=") != head0.end()) {
                if (head0.size() != 2 || head0[1] != "=" || !is_name(head0[0]))
                    throw error(fmt::format("bad fragment definition {}", fmt::join(head0, " ")));
                {
                    facul_method::parameters dummy;
                    if (facul_method_by_name(head0[0], dummy) || head0[0] == "STOP")
                        throw error(fmt::format("{} is a method, not a fragment name", head0[0]));
                }
                if (!fragments.emplace(head0[0], chain_words).second)
                    throw error(fmt::format("fragment {} redefined", head0[0]));
                /* check it now, rather than at its first use */
                expand(fragments.at(head0[0]), 0, 0);
                continue;
            }
            std::string const hs = fmt::format("{}", fmt::join(head0, " "));
            if (hs.empty() || hs.back() != ':')
                throw error(fmt::format("neither a cell \"R0 R1:\" nor a fragment \"name =\": {}", hs));
            auto const head = tokens(hs.substr(0, hs.size() - 1));
            if (head.size() != mfb.size())
                throw error(fmt::format("{} sizes for {} sides", head.size(), mfb.size()));
            std::vector<std::pair<unsigned int, unsigned int>> ranges;
            for (auto const & h: head) {
                auto const dash = h.find('-');
                unsigned int const a = number(h.substr(0, dash));
                unsigned int const b = dash == std::string::npos ? a : number(h.substr(dash + 1));
                if (b < a)
                    throw error(fmt::format("empty range {}", h));
                ranges.emplace_back(a, b);
            }
            chain_t const chain = expand(chain_words, 0, 0);
            std::vector<unsigned int> k(ranges.size());
            std::function<void(size_t)> fill = [&](size_t i) {
                if (i == ranges.size()) {
                    if (chain.empty())
                        parsed.erase(k);
                    else
                        parsed[k] = chain;
                    return;
                }
                for (k[i] = ranges[i].first ; k[i] <= std::min(ranges[i].second, mfb[i]) ; k[i]++)
                    fill(i + 1);
            };
            fill(0);
        } catch (error const & e) {
            throw cado::error("Parse error on line {} of strategies file: {}", lnum, e.what());
        } catch (std::logic_error const & e) {
            throw cado::error("Parse error on line {} of strategies file: {}", lnum, e.what());
        }
    }
    for (auto & [k, chain]: parsed) {
        try {
            parameter_sequence_tracker::fill_default_parameters(chain);
        } catch (error const & e) {
            throw cado::error("Parse error in strategies file while setting parameters for r={}: {}", k, e.what());
        }
    }
    return parsed;
}

static facul_strategies::strategy_file
read_strategy_file(std::vector<unsigned int> const & mfb, FILE * file)
{
    std::string text;
    fseek(file, 0, SEEK_SET);
    for (char buf[4096] ; fgets(buf, sizeof(buf), file) != nullptr ; )
        text += buf;
    return parse_strategy_file(mfb, text);
}

template<class UnaryOp>
void facul_strategies::for_each_sizes_combination(UnaryOp f) const
{
    std::vector<unsigned int> v(mfb.size());
    for_each_sizes_combination_inner(0, v, f);
}

template<class UnaryOp>
void facul_strategies::for_each_sizes_combination_inner(unsigned int i, std::vector<unsigned int> & v, UnaryOp f) const
{
    if (i >= mfb.size()) {
        f(v);
    } else {
        for(size_t j = 0; j <= mfb[i]; ++j) {
            v[i] = j;
            for_each_sizes_combination_inner(i+1, v, f);
        }
    }
}

/* what a method of the book was made from */
static facul_method::parameters_with_side parameters_of(facul_method_side const & ms)
{
    facul_method const & fm = *ms.method;
    facul_method::parameters_with_side m;
    switch (fm.method) {
        case PM1_METHOD: {
            auto const * p = (pm1_plan_t const *) fm.plan;
            m = { ms.side, PM1_METHOD, (unsigned long) p->B1, (unsigned long) p->stage2.B2 };
            break;
        }
        case PP1_27_METHOD:
        case PP1_65_METHOD: {
            auto const * p = (pp1_plan_t const *) fm.plan;
            m = { ms.side, fm.method, (unsigned long) p->B1, (unsigned long) p->stage2.B2 };
            break;
        }
        case EC_METHOD: {
            auto const * e = (ecm_plan_t const *) fm.plan;
            m = { ms.side, EC_METHOD, (unsigned long) e->B1, (unsigned long) e->stage2.B2,
                  e->parameterization, e->parameter, 1 };
            break;
        }
        case MPQS_METHOD:
            m = { ms.side, MPQS_METHOD, 0UL, 0UL };
            break;
        default:
            ASSERT_ALWAYS(0);
    }
    m.stop_unsplit = ms.stop_unsplit;
    return m;
}

/* the book, as a strategy file */
void facul_strategies::print(FILE * file) const
{
    if (file == nullptr)
        return;
    fmt::print(file, "# lpb = {}, mfb = {}\n", lpb, mfb);
    strategy_file S;
    for_each_sizes_combination([&](std::vector<unsigned int> & sizes) {
            auto const & v = (*this)(sizes);
            for (auto const & ms: v)
                S[sizes].push_back(parameters_of(ms));
            });
    std::ostringstream os;
    write_file(os, S);
    fputs(os.str().c_str(), file);
}

facul_strategies_base::facul_strategies_base (
        std::vector<unsigned long> const & lim,
        std::vector<unsigned int> const & lpb,
        std::vector<unsigned int> const & mfb,
        bool perfectly_sieved)
    : B(lim), lpb(lpb), BB(lim.size()), BBB(lim.size()), mfb(mfb)
{
    ASSERT_ALWAYS(lpb.size() == lim.size());
    ASSERT_ALWAYS(mfb.size() == lim.size());
    auto pBB = BB.begin();
    auto pBBB = BBB.begin();
    for(double const b: B) {
        *pBB++ = perfectly_sieved ? b*b : 0;
        *pBBB++ = b*b*b;
    }
}

void facul_strategies::precompute_method(facul_method::parameters const & mp, int verbose)
{
    /* How do I emplace with a guarantee of not calling the ctor
     * needlessly? map::emplace is actually allowed to construct
     * unconditionally.
     */
    auto it = precomputed_methods.emplace(mp, facul_method());
    if (it.second)
        it.first->second = facul_method(mp, verbose);
}

/* Create our strategy book from a file. */
facul_strategies::strategy_file
facul_strategies::read_file(std::vector<unsigned int> const & mfb, FILE * file)
{
    return read_strategy_file(mfb, file);
}

facul_strategies::facul_strategies(
        std::vector<unsigned long> const & lim,
        std::vector<unsigned int> const & lpb,
        std::vector<unsigned int> const & mfb,
        bool perfectly_sieved,
        FILE * file,
        const int verbose)
    : facul_strategies(lim, lpb, mfb, perfectly_sieved, read_strategy_file(mfb, file), verbose)
{
}
 
/* Create our strategy book from an in-memory version of a file. */
facul_strategies::facul_strategies(
        std::vector<unsigned long> const & lim,
        std::vector<unsigned int> const & lpb,
        std::vector<unsigned int> const & mfb,
        bool perfectly_sieved,
        facul_strategies::strategy_file const & parsed_file,
        const int verbose)
    : facul_strategies_base(lim, lpb, mfb, perfectly_sieved)
{
    /* Add all methods to the cache */
    for(auto const & v : parsed_file) {
        auto const & chain_parameters = v.second;
        for(facul_method::parameters const & mp: chain_parameters) {
            precompute_method(mp, verbose);
        }
    }

    /* Add all method chains to the cache */
    for(auto const & v : parsed_file) {
        auto const & chain_parameters = v.second;
        auto it = precomputed_strategies.find(chain_parameters);
        if (it != precomputed_strategies.end())
            continue;
        auto & chain = precomputed_strategies[chain_parameters];
        for(auto const & ms: chain_parameters) {
            chain.emplace_back(&precomputed_methods[ms], ms.side);
            chain.back().stop_unsplit = ms.stop_unsplit;
        }
        facul_method_side::fix_is_last(chain);
    }

    unsigned int M = 1U;
    for (unsigned int m: mfb) {
        M *= (m+1U);
    }
    direct_access.assign(M, nullptr);

    /* fill the direct_access array */
    for(auto const & v : parsed_file) {
        auto const & i = v.first;
        auto const & chain_parameters = v.second;
        auto const * w = &precomputed_strategies.at(chain_parameters);
        direct_access_get(i) = w;
    }
}

std::vector<facul_method_side> const & facul_strategies::operator()(std::vector<unsigned int> const & v) const
{
    if (direct_access.empty()) {
        if (v.size() == 1) /* one side */
            return uniform_strategy[0];
        else if (v.size() == 2) /* two sides */
            return uniform_strategy[v[0] < v[1]];
        else { /* more than two sides */
            std::vector<unsigned int> index(v.size());
            std::iota(index.begin(), index.end(), 0);
            std::ranges::sort(index,
                      [&](const unsigned int& i, const unsigned int& j) {
                          return (v[i] < v[j]);
                      });
            /* index[0] contains the id of the largest side, index[1] the id of
             * the second largest, ...
             * We now need the lexicographic index of this permutation.
             * Note: cost is quadratic in the number of sides.
             * Found the formula at https://stackoverflow.com/questions/12146910/finding-the-lexicographic-index-of-a-permutation-of-a-given-array
             * Adapted it "Horner-style" to avoid the factorials.
             */
            size_t idx = 0;
            for(size_t i = 0; i < index.size()-1; ++i) {
                for(size_t j = i+1; j < index.size(); ++j)
                    idx += (index[j] < index[i] ? 1 : 0);
                idx *= (index.size()-i-1);
            }
            return uniform_strategy[idx];
        }
    } else {
        static std::vector<facul_method_side> const placeholder;
        auto it = direct_access_get(v);
        if (it != nullptr)
            return *it;
        else
            return placeholder;
    }
}

bool facul_strategies::accepts_sizes(std::vector<unsigned int> const & sizes) const
{
    if (!(*this)(sizes).empty())
        return true;
    /* as in facul_all(): without a perfectly sieved factor base, BB is
     * zero, but a prime below B^2 is still smooth */
    for (size_t i = 0; i < sizes.size(); ++i) {
        double const bb = BB[i] ? BB[i] : double(B[i]) * double(B[i]);
        if (sizes[i] > 1 && ldexp(1.0, int(sizes[i]) - 1) >= bb)
            return false;
    }
    return true;
}

std::vector<facul_method_side> const * & facul_strategies::direct_access_get(std::vector<unsigned int> const & v)
{
    ASSERT_ALWAYS(v.size() == mfb.size());
    unsigned int b = 1, idx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        idx += v[i]*b;
        b *= mfb[i]+1;
    }
    return direct_access[idx];
}

std::vector<facul_method_side> const * const & facul_strategies::direct_access_get(std::vector<unsigned int> const &v) const
{
    ASSERT_ALWAYS(v.size() == mfb.size());
    unsigned int b = 1, idx = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        idx += v[i]*b;
        b *= mfb[i]+1;
    }
    return direct_access[idx];
}

/*
  Make a simple strategy for factoring. We start with
  P-1 and P+1 (with x0=2/7), then an ECM curve with low bounds, then
  a bunch of ECM curves with larger bounds. How many methods to do in
  total is controlled by the n parameter: P-1, P+1 and the first ECM
  curve (with small bounds) are always done, then n ECM curves (with
  larger bounds).
  This function is used when you don't give a strategy file.
*/
std::vector<facul_method::parameters>
facul_strategy_oneside::default_strategy (int n)
{
    ASSERT_ALWAYS(n >= 0);

    std::vector<facul_method::parameters> chain;

#if 0
    /* This is relevant only for very weird experiments where we have small
     * factors that stick together.  */
    chain.emplace_back({ PM1_METHOD, 30, 100 });
#endif

    chain.emplace_back(PM1_METHOD, 315, 2205);
    chain.emplace_back(PP1_27_METHOD, 525, 3255);

#ifdef USE_LEGACY_DEFAULT_STRATEGY
    chain.emplace_back(EC_METHOD, 105, 3255, MONTY12, 2, 1);
#else
    // I'm not sure, but it seems that MONTY12 with param=2 (which has
    // nothing special anyway) is not reached by the MONTYTWED12
    // parameterization. Let's pick an arbitrary, other curve from the
    // MONTYTWED12 family, then.
    chain.emplace_back(EC_METHOD, 105, 3255, MONTYTWED12, 2, 1);
#endif

    /* See #30033 ; B1 is not monotonic here, but this is on purpose. The
     * Brent-Suyama curve with sigma=11 (which is isomorphic to the
     * MONTYTWED12 with parameter=1) is exceptional, and yields more
     * primes. Let's put a little bit more effort into it.
     *
     *     sage: load("cofac_utils.sage")
     *     sage: E0,P0=Twed12_parameterization(1, Rationals())
     *     sage: E1,P1=BrentSuyama_parameterization(11, Rationals())
     *     sage: E0.is_isomorphic(E1)
     *     True
     */
    if (n--) {
#ifdef USE_LEGACY_DEFAULT_STRATEGY
        chain.emplace_back(EC_METHOD, 315, 5355, BRENT12, 11, 1);
#else
        chain.emplace_back(EC_METHOD, 315, 5355, MONTYTWED12, 1, 1);
#endif
    }

    /* heuristic strategy where B1 is increased by c*sqrt(B1) at each curve
     *
     * Note that only some addition chains were precomputed in
     * sieve/ecm/bytecode_mishmash_B1_data.h ; those have fewer
     * arithmetic operations than the ones we compute automatically.
     *
     * For this reason, if the sequence of attained values for B1 is
     * changed, the performance mught degrade a little bit.
     */
    double B1 = 105;
    for (int i = 3 ; n-- > 0 ; i++) {
        /* If the sequence of B1 values is modified, it may be a good thing to
         * regenerate bytecode_mishmash_B1_data.h to add precomputed chains for
         * the new B1 values.
         */
        B1 += sqrt (B1);

        /* The factor 50 was determined experimentally with testbench, to find
           factors of 40 bits:
           testbench -p -cof 1208925819614629174706189 -strat 549755813888 549755913888
           This finds 1908 factors (out of 3671 input numbers) with n=24 curves
           and 3.66s.
           With B2=17*B1, and 29 curves, we find 1898 factors in 4.07s.
           With B2=100*B1, and 21 curves we find 1856 factors in 3.76s.
           Thus 50 seems close to optimal.
Warning: changing the value of B2 might break the sieving tests.
*/
        /* we round B2 to (2k+1)*105, thus k is the integer nearest to
           B2/210-0.5 */
        unsigned long const B2 = (2 * (unsigned int) ((50.0 * B1) / 210.0) + 1) * 105;

#ifdef USE_LEGACY_DEFAULT_STRATEGY
        /* FIXME: are MONTY12 and MONTYTWED12 swapped, perhaps ? */
        chain.emplace_back(EC_METHOD, B1, B2, MONTYTWED12, i, 1);
#else
        chain.emplace_back(EC_METHOD, B1, B2, MONTY12, i, 1);
#endif
    }

#ifdef USE_MPQS
    chain.emplace_back({ MPQS_METHOD });
#endif

    return chain;
}

facul_strategy_oneside::facul_strategy_oneside (unsigned long fbb, unsigned int lpb, unsigned int mfb, std::vector<facul_method::parameters> const & mps, int verbose)
    : B(fbb),
      lpb(lpb),
      BB((double) B * (double) B),
      BBB((double) B * (double) B * (double) B),
      mfb(mfb)
{
    for(auto const & mp : mps)
        methods.emplace_back(mp, verbose);
}

facul_strategy_oneside::facul_strategy_oneside (unsigned long fbb, unsigned int lpb, unsigned int mfb, int n, int verbose)
    : facul_strategy_oneside(fbb, lpb, mfb, default_strategy(n < 0 ? nb_curves_with_fbb (fbb, lpb, mfb) : n), verbose)
{}


/*
 * Create our strategy book with our default strategy.
 */
facul_strategies::facul_strategies (
        std::vector<unsigned long> const & lim,
        std::vector<unsigned int> const & lpb,
        std::vector<unsigned int> const & mfb,
        std::vector<int> ncurves,
        bool perfectly_sieved,
	const int verbose)
    : facul_strategies_base(lim, lpb, mfb, perfectly_sieved)
{
    int const nsides = lim.size();

    int max_ncurves = -1;
    for(int side = 0 ; side < nsides ; side++) {
        if (ncurves[side] < 0)
            ncurves[side] = nb_curves_with_fbb (B[side], lpb[side], mfb[side]);
        max_ncurves = std::max(max_ncurves, ncurves[side]);
    }

    verbose_output_start_batch();
    verbose_fmt_print(0, 2, "# Using default strategy for the cofactorization:");
    for (unsigned int i = 0; i < ncurves.size(); i++) {
        verbose_fmt_print(0, 2, " ncurves{}={}", i, ncurves[i]);
    }
    verbose_fmt_print(0, 2, "\n");
    verbose_output_end_batch();

    /* prepare the chain of methods that we want to use in order to
     * factor a number, irrespective of its side.
     */
    auto chain_parameters = facul_strategy_oneside::default_strategy(max_ncurves);

    /* We now need to truncate the list of strategies according to
     * ncurves[i] for 0 <= i < nsides
     *
     * NOTE: This is incompatible with USE_MPQS
     */

    std::vector<std::vector<facul_method::parameters_with_side>> w;

    std::vector<unsigned int> p(nsides);
    std::iota(p.begin(), p.end(), 0);
    do
    {
        /* We iterate over all permutations of [0..nsides-1] in lexicographic
         * order starting from (0,1,...,nsides-1) and using
         * std::next_permutation.
         * Each permutation p corresponds to the case where side p[0] is the
         * largest, side p[1] the second largest, etc..., i.e.,
         *   side p[0] >= side p[1] >= ... >= side p[nsides-1]
         *
         * See the implementation of facul_strategies::operator() to see how the
         * permutation is computed from the sizes of the sides.
         */
        w.emplace_back();
        for(unsigned int side: p) {
            int n = ncurves[side] + (chain_parameters.size() - max_ncurves);
            for(facul_method::parameters const & mp: chain_parameters) {
                if (!n--)
                    break;
                w.back().emplace_back(side, mp);
            }
        }

        /* facul_strategy_oneside::default_strategy is allowed to use
         * default parameters, of course. And it has to abide by the same
         * rules as the strategy files. */
        parameter_sequence_tracker::fill_default_parameters(w.back());
    } while (std::next_permutation(p.begin(), p.end()));

    /* Add all methods to the cache.
     *
     * Note that at least in the present setting, where the method cache
     * is indexed by something that also contains the parameter and so
     * on, we may end up computing an addition chain twice if it so
     * happens that the same B1 is used with two different parameters, or
     * two different EC parameterizations. That is slightly annoying.
     */
    for(auto const & wi: w) {
        for(auto const & mp: wi)
            precompute_method(mp, verbose);
    }

    for(auto const & wi: w) {
        uniform_strategy.emplace_back();
        std::vector<facul_method_side> & cur = uniform_strategy.back();
        for(auto const & mp: wi)
            cur.emplace_back(&precomputed_methods[mp], mp.side);
        facul_method_side::fix_is_last(cur);
    }
}
