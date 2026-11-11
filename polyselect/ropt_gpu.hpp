#ifndef CADO_ROPT_GPU_HPP
#define CADO_ROPT_GPU_HPP

/* GPU root sieve for ropt_stage2.cpp (see ropt_gpu_kernels.h): the lines
 * that rootsieve_run_line would sieve are recorded, then applied on the
 * GPU. CADO_ROPT_GPU=0 turns it off; CADO_ROPT_GPU_CHECK=1 also sieves on
 * the CPU and compares; CADO_ROPT_GPU_VERBOSE=1 prints counts at exit. */

#include <cstddef>
#include <cstdint>

struct ropt_sieve_oplist;

/* records the line j, j + pe, ... <= bound (j <= bound) */
void ropt_sieve_oplist_add (ropt_sieve_oplist *, long j, long bound,
                            unsigned int pe, int16_t sub);

/* an empty op list if the GPU sieves an array of len entries, else
   NULL */
ropt_sieve_oplist * ropt_gpu_begin (size_t len);

/* the sieve array: init minus the recorded lines */
void ropt_gpu_finish (ropt_sieve_oplist *, int16_t init, int16_t * array,
                      size_t len);

/* get_alpha's root counts for these polynomials (f, the algebraic side)
   on the GPU: from here until ropt_gpu_alpha_end, get_alpha on any of
   them uses the counts (see set_alpha_rootcount_source). Returns false
   if the GPU is not used. */
#include <vector>
#include "cxx_mpz.hpp"
#include "mpz_poly.h"
bool ropt_gpu_alpha_begin (std::vector<cxx_mpz_poly> const & fs);
void ropt_gpu_alpha_end ();

#endif
