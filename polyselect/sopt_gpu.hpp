#ifndef CADO_SOPT_GPU_HPP
#define CADO_SOPT_GPU_HPP

/* Local descents of size optimization on the GPU (sopt_gpu_kernels.cu).
 *
 * Used when the GPU has fast double precision (FP32/FP64 throughput ratio
 * at most 4: H100, B200 and the like); on other GPUs the CPU is faster.
 * CADO_POLYSELECT_GPU_SOPT=0 always uses the CPU, =1 always the GPU (if
 * there is one); CADO_POLYSELECT_GPU=0 turns all of polyselect's GPU code
 * off. CADO_POLYSELECT_GPU_VERBOSE=1 prints counts at exit.
 */

#include <vector>
#include "cxx_mpz.hpp"
#include "mpz_poly.h"

/* whether sopt_gpu_descents uses the GPU (in this thread) */
bool sopt_gpu_enabled ();

/* sopt_local_descent (f_opt, g_opt, f[i], g[i], 1, deg_rotation,
 * max_iter, 0) for every i, f[i] and g[i] replaced by f_opt and g_opt,
 * lognorm[i] its return value. The GPU runs the descents; those it gives
 * back are finished (or redone) on the CPU. Returns false, having done
 * nothing, if the GPU is not used. */
bool sopt_gpu_descents (std::vector<cxx_mpz_poly> & f,
                        std::vector<cxx_mpz_poly> & g, int deg_rotation,
                        unsigned int max_iter, std::vector<double> & lognorm);

#endif
