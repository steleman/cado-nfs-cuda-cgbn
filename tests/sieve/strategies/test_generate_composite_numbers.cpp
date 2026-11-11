#include "cado.h" // IWYU pragma: keep
#include "cofactoring_method_bench.hpp"

#include <cstdlib>
#include <cstdio>
#include <vector>

#include "gmp_aux.h"
#include "random_distributions.hpp"
#include "cofactoring_method.hpp"

/*
  This program tests the following functions:
  - distribution_prime_number()              (in generate_factoring_method.cpp)
  - generate_prime_factor()                  (in generate_factoring_method.cpp)
  - generate_composite_integer()             (in generate_factoring_method.cpp)
  - generate_composite_integer_interval()    (in generate_factoring_method.cpp)
*/


int main ()
{
    cxx_gmp_randstate state;

    //check dist
    double precision = 0.00001;
    int len_p_min = 10;
    int len_p_max = 10;
    auto dist = distribution_prime_number(len_p_min, len_p_max);
    if (dist[0] > (1+precision) || dist[0] < (1-precision))
	{
	    fprintf (stderr, "error with the test(1)\n");
	    return EXIT_FAILURE;
	}
    
    len_p_min = 10;
    len_p_max = 14;
    dist = distribution_prime_number(len_p_min, len_p_max);
    double check_sum = dist[0];
    for (int i = 0; i < 4; i++)
	{
	    if (dist[i] < dist[i+1])
		{
		    fprintf (stderr, "error with the test(2)\n");
			    return EXIT_FAILURE;
		}
	    check_sum+=dist[i+1];
	}
    if (check_sum > 1+precision || check_sum < 1-precision)
	{
	    fprintf (stderr, "error with the test(3)\n");
            fprintf (stderr, "check_sum=%f precision=%f\n",
                     check_sum, precision);
	    return EXIT_FAILURE;
	}

    //Test the random selection of our size according the distribution 'dist'!
    precision = 0.02;
    int len = len_p_max-len_p_min+1;
    std::vector<double> dist_test(len, 0);
    int nb_test = 10000;
    for (int i = 0; i < nb_test; i++)
	{
	    unsigned int index = random_categorical(state, dist);
	    dist_test[index]+=1;
	}
    //check the similarity!!!
    for (int i = 0; i < len; i++)
	if ((dist_test[i]/nb_test - dist[i]) > precision ||
	    (dist[i] - dist_test[i]/nb_test) > precision)
	    {
		fprintf (stderr, "error with the test(4)\n");
                for (int j = 0; j < len; j++)
                  fprintf (stderr, "dist[%d]=%f dist_test[%d]=%f\n",
                           j, dist[j], j, dist_test[j]);
                fprintf (stderr, "dist_test[i]/nb_test - dist[i]=%f\n",
                         dist_test[i]/nb_test - dist[i]);
                fprintf (stderr, "precision=%f\n", precision);

		return EXIT_FAILURE;
	    }
    return EXIT_SUCCESS;
}
