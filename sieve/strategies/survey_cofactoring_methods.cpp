#include "cado.h" // IWYU pragma: keep

#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>

#include <vector>

#include <gmp.h>

#include "gmp_aux.h"
#include "params.hpp"
#include "facul_ecm.h"
#include "facul_method.hpp"
#include "fmt/format.h"
#include "facul_strategies.hpp"
#include "cofactoring_method_bench.hpp"
#include "macros.h"
#include "cado_main.hpp"
#include "utils_cxx.hpp"
#include "cofactoring_method.hpp"
#include "convex_hull.hpp"

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header(fmt::format(
                            "This binary allows to choose the best parameters (B1, B2 (B2=c*B1))\n"
			    "for each factoring methods: {},\n"
			    "to find a prime number in [2^lb, 2^ub]. Note that you can : \n"
			    "-make your study only with one method (-m).\n"
			    "-specify the sieving region of B1 and c instead of using default values.\n"
			    "Then, You can apply the convex hull to keep only the best methods\n "
			    "(-fch from a file or just -ch if you want to apply it after the bench\n).",
                            facul_method_family_names()));
    pl.declare_usage("ub", "to set large prime bound to 2^ub.");
    pl.declare_usage("lb", "to set factor base bound to 2^lb.");
    pl.declare_usage("n", "to give the length of n.");
    pl.declare_usage("out",
			  "to specify the file which contain our factoring methods.");
    pl.declare_usage("m",
			  fmt::format("to specify the method, among: {}.\n"
                              "By default, we use all methods one after the other.",
                              facul_method_family_names()));
    pl.declare_usage("ch", "to apply the convex hull.");
    pl.declare_usage("seed", "random seed");

    pl.declare_usage("b1min", "to set b1_min (sieve region).");
    pl.declare_usage("b1max", "to set b1_max (sieve region).");
    pl.declare_usage("b1step", "to set b1_step (sieve region).");
    pl.declare_usage("cmin", "to set c_min (sieve region).");
    pl.declare_usage("cmax", "to set c_max (sieve region).");
    pl.declare_usage("cstep", "to set c_step (sieve region).");

    pl.declare_usage("fch", "to apply the convex hull");
    pl.declare_usage("fch_in",
			  "to specify the input file which contains \n the factoring methods.");
    pl.declare_usage("fch_out",
			  "to specify the output file which contains \n the convex hull.");

}

/************************************************************************/
/*                      MAIN */
/************************************************************************/

static int main_(int argc, char const * argv[]);

// coverity[root_function]
int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;

    declare_usage(pl);

    pl.configure_switch("ch");
    pl.configure_switch("fch");

    pl.process_command_line_and_extra_parameter_files(argc, argv);

    int const opt_fch = pl.parse<int>("-fch");
    if (opt_fch) {
	auto const pathname_fch_in = pl.parse_mandatory<std::string>("fch_in");
	auto const pathname_fch_out = pl.parse_mandatory<std::string>("fch_out");

        std::ifstream file_in(pathname_fch_in);
        if (!file_in)
            throw cado::error("impossible to read {}", pathname_fch_in);
        std::ofstream file_out(pathname_fch_out);
        if (!file_out)
            throw cado::error("impossible to write in the file {}",
                    pathname_fch_out);

	convex_hull_from_file(file_in, file_out);
    } else {
	//default values
	int lb = -1, ub = -1, len_n = -1;
        ec_parameterization_t curve = MONTY12;
	// {b1min, b1max, b1step, cmin, cmax, cstep}
        std::vector<int> param(6, 0);

	int const opt_ch = pl.parse<int>("-ch");
	pl.parse("ub", ub);
	pl.parse("lb", lb);
	pl.parse("n", len_n);
	pl.parse("b1min", param[0]);
	pl.parse("b1max", param[1]);
	pl.parse("b1step", param[2]);
	pl.parse("cmin", param[3]);
	pl.parse("cmax", param[4]);
	pl.parse("cstep", param[5]);

	if (lb == -1 || ub == -1 || ub <= lb)
	    pl.fail("options -lb, -ub are mandatory here,\n"
		    "and lb must be less than ub.\n\n");

	/*
	   Default value for len_n:
	   So n will be a composite such that n=pq, with :
	   size of p equal to len_p bits and for q 3*len_p bits.
	   Thus, q will not be a problem when we are going to p!
	 */
	if (len_n == -1)
	    len_n = 60 + ub;

	/* Extract the method and the curve\n" */
	std::string name_fm;

        facul_method_code method = NO_METHOD;

	if (pl.parse("m", name_fm)) {
            /* The families and their names are the ones a strategy file
             * uses, and they are read back with the same function, so a
             * name accepted here is a name las can read. */
            facul_method::parameters family;
            if (!facul_method_by_name(name_fm, family))
		pl.fail("Your factoring method '{}' doesn't exist.\nChoose another method among: {}",
		        name_fm, facul_method_family_names());
            method = family.method;
            if (method == EC_METHOD)
                curve = family.parameterization;
	}

	cxx_gmp_randstate state;
	unsigned long seed = 0;
	if (pl.parse("seed", seed))
	    gmp_randseed_ui(state, seed);

	/*
	   To generate our factoring methods!
	 */

	int *param_sieve = nullptr;
	if (param[0] && param[1] && param[2] &&
	    param[3] && param[4] && param[5])
        {
	    param_sieve = param.data();
            /*
	    printf("param_sieve: b1min= %d, b1max=%d, b1step=%d"
		   "       cmin= %d, cmax=%d, cstep=%d\n",
		   param_sieve[0], param_sieve[1], param_sieve[2],
		   param_sieve[3], param_sieve[4], param_sieve[5]);
                   */
	}

	cofactoring_method_list const res = (method == NO_METHOD)
	    ? generate_factoring_methods
		(state, lb, ub, len_n, opt_ch, param_sieve)
	    : generate_factoring_methods_mc
		(state, lb, ub, len_n, method, curve, opt_ch, param_sieve);

	//print the result in file!
	//to do: change the file output!
        /* if -out is not given, print to stdout */
	std::string name_file_out;
        if (!pl.parse("out", name_file_out)) {
            std::cout << res;
        } else {
            std::ofstream file_out(name_file_out);
            if (!file_out || !(file_out << res))
                throw cado::error("error:: try to write in the file '{}'.",
                        name_file_out);
        }
    }

    /* An option neither branch reads is a mistake worth reporting, not
     * something to ignore silently. */
    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    return EXIT_SUCCESS;
}
