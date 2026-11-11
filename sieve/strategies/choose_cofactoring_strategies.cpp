#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <fstream>

#include "select_best_strategies.hpp"
#include "params.hpp"
#include "cofactoring_strategy.hpp"
#include "macros.h"
#include "cado_main.hpp"
#include "utils_cxx.hpp"

/************************************************************************/
/*             USAGE                                                    */
/************************************************************************/

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header("This binary allows to choose the good strategy for each pair\n"
			    "\t\t of cofactor according to the distribution of these pairs\n"
			    "\t\t and the time required to establish our cofactors.\n");
    pl.declare_usage("st",
			  "the pathname of our directory which contains our strategies.");
    pl.declare_usage("dist",
			  "the pathname of our file which contains the distribution\n"
			  "\t\t of our pairs of cofactors.");
    pl.declare_usage("t",
			  "specify the time (seconds) to obtain cofactors in the file\n"
			  "\t\t given by the option 'dist'.");
    pl.declare_usage("mfb0", "set the first cofactor bound to 2^mfb0");
    pl.declare_usage("mfb1",
			  "set the second cofactor bound to 2^mfb1");
    pl.declare_usage("out",
			  "the output file which contain final strategies.");

}

/************************************************************************/
/*             MAIN                                                     */
/************************************************************************/

static int main_(int argc, char const * argv[]);

int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;
    declare_usage(pl);

    pl.process_command_line_and_extra_parameter_files(argc, argv);


    int mfb0 = -1, mfb1 = -1;
    double time_C = -1;
    pl.parse("mfb0", mfb0);
    pl.parse("mfb1", mfb1);
    pl.parse("t", time_C);
    if (mfb0 < 0 || mfb1 < 0 || time_C < 0)
	pl.fail("The following parameters are mandatory:\n"
	      "\t\t -mfb0 -mfb1 -t\n");

    printf("time_C = %lf seconds!\n", time_C);
    //convert the time in micro-s. because all previous binaries
    //compute their times in micro-s.
    time_C *= 1000000;		//s-->micro-s

    auto const pathname_C = pl.parse_mandatory<std::string>("dist");
    auto const pathname_st = pl.parse_mandatory<std::string>("st");
    auto const pathname_output = pl.parse_mandatory<std::string>("out");

    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    printf("EXTRACT DATA STRAT\n");
    auto const matrix_strat =
	extract_matrix_strat(pathname_st, mfb0 + 1, mfb1 + 1);

    printf("EXTRACT DATA C\n");
    std::ifstream file_C(pathname_C);
    if (!file_C)
	throw cado::error("Error while reading file {}", pathname_C);
    auto const matrix_C = extract_matrix_C(file_C, mfb0 + 1, mfb1 + 1);

    printf("GENERATE FINAL STRATEGIES\n");
    auto const matrix_strat_res =
	compute_best_strategy(matrix_strat, matrix_C, mfb0 + 1, mfb1 + 1,
			      time_C);

    std::ofstream file_output(pathname_output);
    if (!file_output)
	throw cado::error("Error when i want to write in '{}'", pathname_output);
    fprint_final_strategy(file_output, matrix_strat_res, mfb0 + 1, mfb1 + 1);

    return EXIT_SUCCESS;
}
