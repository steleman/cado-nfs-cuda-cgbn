#include "cado.h" // IWYU pragma: keep

#include <cstdlib>

#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "fmt/base.h"

#include "cado_main.hpp"
#include "factorization_pattern.hpp"
#include "factorization_patterns.hpp"
#include "params.hpp"
#include "cofactoring_strategy.hpp"

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header(
        "List the ways a cofactor can factor, by the sizes of its prime "
        "factors.\n"
        "\n"
        "For a cofactor of mfb bits none of whose prime factors is below "
        "lim, this\n"
        "prints one line per pattern of factor sizes, with an estimate of "
        "how many\n"
        "such cofactors factor that way:\n"
        "\n"
        "    [ 20 21 ]: 1.12209e+09\n"
        "\n"
        "means that about 1.12e9 of them are the product of a 20-bit and a "
        "21-bit\n"
        "prime. This is what the strategy generator reads to know which "
        "prime sizes\n"
        "its strategies have to be able to find;\n"
        "\"build_cofactoring_strategies -patterns\" writes the same thing.\n");

    pl.declare_usage("lim", "factor base bound: no prime factor is below this");
    pl.declare_usage("mfb", "size of the cofactor, in bits");
    pl.declare_usage("out", "write here instead of to standard output");
    pl.declare_usage("compare",
                     "print the counts computed both ways -- by the "
                     "generating series\n"
                     "\t\t and by sampling -- with their ratio, instead of "
                     "the list");
    pl.declare_usage("seed", "random seed, used by -compare only");
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;
    declare_usage(pl);
    pl.configure_switch("compare");

    pl.process_command_line_and_extra_parameter_files(argc, argv);

    unsigned long lim = 0;
    int mfb = -1;
    unsigned long seed = 0;
    pl.parse("lim", lim);
    pl.parse("mfb", mfb);
    pl.parse("seed", seed);

    if (lim == 0)
        pl.fail("Error: parameter -lim is mandatory\n");
    if (mfb < 0)
        pl.fail("Error: parameter -mfb is mandatory\n");

    std::string file_out;
    std::ofstream out_file;
    bool const to_file = pl.parse("out", file_out);
    if (to_file) {
        out_file.open(file_out);
        if (!out_file)
            throw cado::error("Cannot write {}", file_out);
    }
    std::ostream & out = to_file ? out_file : std::cout;

    bool const compare = pl.parse<int>("-compare");

    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    if (compare)
        compare_pattern_estimates(mfb, lim, seed, out);
    else
        out << list_factorization_patterns(mfb, lim);

    return EXIT_SUCCESS;
}

// coverity[root_function]
int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}
