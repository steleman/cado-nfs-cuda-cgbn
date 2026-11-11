#ifndef CADO_LAS_SIEVE_GPU_HPP
#define CADO_LAS_SIEVE_GPU_HPP

/* GPU siever, see las-sieve-gpu.cpp. */

#include <cstdint>

#include <vector>

#include "bucket.hpp"

class nfs_work;
struct qlattice_basis;

/* What process_bucket_region needs of one bucket region once the GPU
 * has sieved: the survivors (in increasing order) with their sieve
 * values, and what purge_buckets and resieve would have found on them. */
struct gpu_sieve_region {
    unsigned int N = 0;
    std::vector<uint16_t> x;
    std::vector<uint8_t> S[2];
    std::vector<bucket_update_t<1, longhint_t>> purged[2];
    std::vector<bucket_update_t<1, primehint_t>> primes[2];
};

/* Whether this special-q is sieved on the GPU: a GPU, two sides with
 * factor bases, no sublattices, logI <= 16, and not CADO_LAS_GPU_SIEVE=0. */
bool las_sieve_gpu_enabled(nfs_work const & ws, qlattice_basis const & Q);

/* After small_sieve_init on both sides: sieves on the GPU and gives the
 * regions that have survivors. */
/* With wc (and its accept_estimate not empty), the survivors that
 * filter_survivors_by_region would drop are dropped on the GPU, before
 * resieving and copying. */
class nfs_work_cofac;
bool las_sieve_gpu_sieve(nfs_work & ws, qlattice_basis const & Q,
        std::vector<gpu_sieve_region> & regions,
        nfs_work_cofac const * wc = nullptr);

/* Check mode (CADO_LAS_GPU_SIEVE_CHECK=1): las sieves on the CPU, and
 * process_bucket_region compares its sieve arrays with the GPU's. */
bool las_sieve_gpu_run(nfs_work & ws, qlattice_basis const & Q);
void las_sieve_gpu_check_region(nfs_work const & ws, int side, unsigned int N,
        unsigned char const * S);

void las_sieve_gpu_report();

#endif /* CADO_LAS_SIEVE_GPU_HPP */
