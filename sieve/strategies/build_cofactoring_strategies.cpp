#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstdlib>

#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "fmt/base.h"

#include "facul_ecm.h"
#include "facul_method.hpp"
#include "facul_strategies.hpp"
#include "cofactoring_method.hpp"
#include "build_strategies.hpp"
#include "macros.h"
#include "params.hpp"
#include "cado_main.hpp"
#include "cofactoring_strategy.hpp"

/************************************************************************/
/*                            USAGE                                     */
/************************************************************************/

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header(
        "Build the best cofactoring strategies for each pair of cofactor "
        "sizes.\n"
        "\n"
        "A strategy is a chain of factoring methods to run on a pair of "
        "cofactors of\n"
        "r0 and r1 bits. This reads the benchmarked methods that "
        "bench_cofactoring_methods\n"
        "produced, and for each pair of sizes keeps the chains that are not "
        "beaten on\n"
        "both counts at once -- more likely to find a factor, and faster. "
        "Which one of\n"
        "those to actually use is then chosen by "
        "choose_cofactoring_strategies.\n"
        "\n"
        "What -decomp names is written by list_factorization_patterns.\n"
        "\n"
        "With no mode switch, the whole matrix is computed in one go, and "
        "-decomp\n"
        "names a directory. The work can also be split in two, which is "
        "what the\n"
        "switches below are for: -oneside for each size of each side, then "
        "-merge to\n"
        "put the two sides together.\n");

    pl.declare_usage("oneside",
        "(mode) keep the best strategies for one cofactor size on one "
        "side.\n"
        "\t\t Needs -lim0, -lpb0, -r0, -ncurves, -decomp.\n");
    pl.declare_usage("merge",
        "(mode) put together what -oneside precomputed for the two sides, "
        "for\n"
        "\t\t one pair (-r0 -r1) or for every pair (-mfb0 -mfb1).\n"
        "\t\t Needs -lim0, -lim1, -in.\n");

    pl.declare_usage("ncurves", "how many ECM curves a strategy may chain");
    pl.declare_usage("r0", "bit size of the cofactor on side 0");
    pl.declare_usage("r1", "bit size of the cofactor on side 1");
    pl.declare_usage("lim0", "factor base bound on side 0");
    pl.declare_usage("lim1", "factor base bound on side 1");
    pl.declare_usage("lpb0", "large prime bound on side 0, in bits");
    pl.declare_usage("lpb1", "large prime bound on side 1, in bits");
    pl.declare_usage("mfb0", "cofactor bound on side 0, in bits");
    pl.declare_usage("mfb1", "cofactor bound on side 1, in bits");
    pl.declare_usage("in",
        "the file of benchmarked methods, or -- with -merge -- the "
        "directory\n"
        "\t\t where -oneside left its results");
    pl.declare_usage("out", "directory to write the results in");
    pl.declare_usage("decomp",
        "the factorization patterns: one file with -oneside, a directory "
        "of\n"
        "\t\t them (named decomp_<lim>_<r>) otherwise");
}

/************************************************************************/
/*     MAIN                                                             */
/************************************************************************/

namespace
{

/* Everything build_cofactoring_strategies reads and writes goes through these, so that the file
 * names stay in one place. */
std::string oneside_filename(std::string const & dir, unsigned long lim, int r)
{
    return fmt::format("{}/strategies{}_{}", dir, lim, r);
}

std::string pair_filename(std::string const & dir, int r0, int r1)
{
    return fmt::format("{}/strategies_{}_{}", dir, r0, r1);
}

template <typename T> T read_from(std::string const & filename)
{
    std::ifstream is(filename);
    T res;
    if (!is || !(is >> res))
        throw cado::error("Cannot read {}", filename);
    return res;
}

template <typename T> void write_to(std::string const & filename, T const & x)
{
    std::ofstream os(filename);
    if (!os || !(os << x))
        throw cado::error("Cannot write {}", filename);
}

/* One methods file, split into the three lists the strategy generator
 * chains together: P-1, P+1 (both flavours at once), and ECM (every
 * parameterization at once).
 *
 * The families are the ones facul names, so a parameterization added
 * there is picked up here; each of them must appear in the file, since
 * each is indexed at [0] and handed to the convex hull, and an empty one
 * is a user error rather than something to paper over. */
struct method_families {
    cofactoring_method_list pm1;
    cofactoring_method_list pp1;
    cofactoring_method_list ecm;

    method_families(cxx_param_list & pl, std::string const & filename)
    {
        auto const all = read_from<cofactoring_method_list>(filename);

        /* Every family needs a zero method, so that the generator can
         * leave a slot empty. */
        auto const zero = cofactoring_method::from_fields(0, 0, 0, 0);
        auto with_zero = [&zero](cofactoring_method_list t) {
            if (t.empty() || t[0].params.B1 != 0)
                t.push_back(zero);
            return t;
        };

        /* What counts as one family here: each ECM parameterization is
         * its own, and gets its own zero method, but the two P+1
         * flavours are pooled, so a file with only one of them is fine.
         * That is what test_missing_method_family.sh checks. */
        std::string missing;
        for (auto const & f: facul_method_families()) {
            /* the curve slot of a non-ECM method carries no information */
            auto const para =
                f.method == EC_METHOD ? f.parameterization : BRENT12;
            auto one = extract_fm_method(all, f.method, para);

            if (f.method == PP1_27_METHOD || f.method == PP1_65_METHOD) {
                pp1.insert(pp1.end(), one.begin(), one.end());
                continue;
            }
            if (one.empty()) {
                missing = facul_method_name(f);
                break;
            }
            one = with_zero(std::move(one));
            if (f.method == PM1_METHOD)
                pm1 = std::move(one);
            else
                ecm.insert(ecm.end(), one.begin(), one.end());
        }
        if (missing.empty() && pp1.empty())
            missing = "PP1-27 or PP1-65";
        if (!missing.empty())
            pl.fail("Error: file {} contains no {} method\n", filename,
                    missing);

        pp1 = with_zero(std::move(pp1));

        sort_by_proba(pm1);
        sort_by_proba(pp1);
        sort_by_proba(ecm);
    }
};

/* -oneside: the best strategies for one cofactor size on one side. */
void subcommand_oneside(cxx_param_list & pl, std::string const & dir_out,
                      method_families const & fam, unsigned long lim0,
                      int lpb0, int ncurves)
{
    int r0 = -1;
    pl.parse("r0", r0);
    if (r0 == -1 || lpb0 == -1 || lim0 == 0 || ncurves == -1)
        pl.fail("Error: parameters -r0 -lim0 -lpb0 -ncurves are mandatories.\n");

    /* Read -decomp whether or not this size needs it: a caller looping
     * over every size passes it every time, and cannot be expected to
     * know which sizes are below the threshold. */
    std::string pattern_file;
    bool const have_pattern_file = pl.parse("decomp", pattern_file);

    unsigned int const fbb0 = ceil(log2((double)(lim0 + 1)));
    factorization_pattern_list patterns;
    if ((unsigned int)r0 >= 2 * fbb0 - 1) {
        if (!have_pattern_file)
            pl.fail("missing mandatory argument decomp\n");
        patterns = read_from<factorization_pattern_list>(pattern_file);
    }

    auto const zero = cofactoring_method::from_fields(0, 0, 0, 0);
    auto const res =
        generate_strategies_oneside(patterns, zero, fam.pm1, fam.pp1,
                                    fam.ecm, ncurves, lim0, lpb0, r0);

    write_to(oneside_filename(dir_out, lim0, r0), res);
}

/* -merge: merge the one-sided precomputations of the two sides. */
void subcommand_merge(cxx_param_list & pl, std::string const & dir_out,
                    unsigned long lim0, unsigned long lim1, int mfb0, int mfb1)
{
    auto const dir_in = pl.parse_mandatory<std::string>("in");
    if (lim0 == 0)
        pl.fail("Error: parameter -lim0 is mandatory\n");
    if (lim1 == 0)
        pl.fail("Error: parameter -lim1 is mandatory\n");

    int r0 = -1, r1 = -1;
    pl.parse("r0", r0);
    pl.parse("r1", r1);

    if (r0 != -1 && r1 != -1) {
        // just one pair
        auto const strat_r0 =
            read_from<cofactoring_strategy_list>(oneside_filename(dir_in, lim0, r0));
        auto const strat_r1 =
            read_from<cofactoring_strategy_list>(oneside_filename(dir_in, lim1, r1));
        write_to(pair_filename(dir_out, r0, r1),
                 generate_strategy_r0_r1(strat_r0, strat_r1));
        return;
    }

    if (mfb0 == -1)
        pl.fail("Error: parameter -mfb0 is mandatory\n");
    if (mfb1 == -1)
        pl.fail("Error: parameter -mfb1 is mandatory\n");

    std::vector<cofactoring_strategy_list> data_rat(mfb0 + 1);
    for (int r = 0; r <= mfb0; r++)
        data_rat[r] =
            read_from<cofactoring_strategy_list>(oneside_filename(dir_in, lim0, r));

    for (int b = 0; b <= mfb1; b++) {
        auto const strat_r1 =
            read_from<cofactoring_strategy_list>(oneside_filename(dir_in, lim1, b));
        for (int a = 0; a <= mfb0; a++)
            write_to(pair_filename(dir_out, a, b),
                     generate_strategy_r0_r1(data_rat[a], strat_r1));
    }
}

/* No switch: do the whole matrix without any precomputation. */
void subcommand_matrix(cxx_param_list & pl, std::string const & dir_out,
                       method_families const & fam, unsigned long lim0,
                       int lpb0, int mfb0, unsigned long lim1, int lpb1,
                       int mfb1, int ncurves)
{
    if (lim0 == 0 || lpb0 == -1 || mfb0 == -1 || lim1 == 0 || lpb1 == -1 ||
        mfb1 == -1 || ncurves == -1)
        pl.fail("Error: parameters -lim0 -lpb0 -mfb0"
                " -lim1 -lpb1 -mfb1 -ncurves are mandatories\n");

    char const * pattern_dir = pl.lookup_old("decomp");
    if (pattern_dir == nullptr)
        pl.fail("Error: parameter -factorization_pattern is mandatory\n");

    auto const matrix =
        generate_matrix(pattern_dir, fam.pm1, fam.pp1, fam.ecm, ncurves, lim0, lpb0,
                        mfb0, lim1, lpb1, mfb1);

    for (int r0 = 0; r0 <= mfb0; r0++)
        for (int r1 = 0; r1 <= mfb1; r1++)
            write_to(pair_filename(dir_out, r0, r1), matrix[r0][r1]);
}

} // namespace

static int build_cofactoring_strategies(int argc, char const * argv[])
{
    cxx_param_list pl;
    declare_usage(pl);

    pl.configure_switch("merge");
    pl.configure_switch("oneside");

    pl.process_command_line_and_extra_parameter_files(argc, argv);

    unsigned long lim0 = 0;
    unsigned long lim1 = 0;
    int mfb0 = -1, mfb1 = -1;
    int lpb0 = -1, lpb1 = -1;
    int ncurves = -1;

    pl.parse("lim0", lim0);
    pl.parse("lim1", lim1);
    pl.parse("lpb0", lpb0);
    pl.parse("lpb1", lpb1);
    pl.parse("mfb0", mfb0);
    pl.parse("mfb1", mfb1);
    pl.parse("ncurves", ncurves);

    auto const dir_out = pl.parse_optional<std::string>("out", "./");

    if (pl.parse<int>("-merge")) {
        subcommand_merge(pl, dir_out, lim0, lim1, mfb0, mfb1);
        /* An option this mode does not read is a mistake worth reporting,
         * not something to ignore silently. */
        if (pl.warn_unused())
            pl.fail("Not all arguments were parsed");
        return EXIT_SUCCESS;
    }

    method_families const fam(pl, pl.parse_mandatory<std::string>("in"));

    if (pl.parse<int>("-oneside"))
        subcommand_oneside(pl, dir_out, fam, lim0, lpb0, ncurves);
    else
        subcommand_matrix(pl, dir_out, fam, lim0, lpb0, mfb0, lim1, lpb1, mfb1,
                          ncurves);

    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    return EXIT_SUCCESS;
}

/* Bad parameters and bad input files are reported by an exception;
 * cado::main_wrapper is what turns it into a message on stderr and a
 * nonzero exit status, rather than whatever the C++ runtime prints
 * before it aborts -- which differs between libstdc++ and libc++, and
 * can be nothing at all.
 */
// coverity[root_function]
int main(int argc, char const * argv[])
{
    return cado::main_wrapper(build_cofactoring_strategies, argc, argv);
}
