#ifndef CADO_LAS_PROCESS_BUCKET_REGION_HPP
#define CADO_LAS_PROCESS_BUCKET_REGION_HPP

#include <memory>

#include "las-where-am-i-proxy.hpp"
#include "las-auxiliary-data.hpp"
#include "sieve-methods.hpp"

/* Ideally this #ifdef should only be in las.ccp and not here */
#ifdef SIQS_SIEVE
    using ALGO = SIQS;
#else
    using ALGO = NFS;
#endif

#include <vector>

class nfs_work;
class nfs_work_cofac;
class thread_pool;
struct gpu_sieve_region;

void process_many_bucket_regions(
        nfs_work & ws,
        std::shared_ptr<nfs_work_cofac> wc_p,
        std::shared_ptr<nfs_aux> aux_p,
        ALGO::special_q_data const & Q,
        thread_pool & pool,
        int first_region0_index,
        where_am_I & w);

/* the regions that the GPU siever found survivors in (las-sieve-gpu.hpp):
 * one task each, queued on the pool, in the group tg if not NULL. The
 * tasks keep their own copy of Q; ws must live until they are done. */
void process_gpu_regions(
        nfs_work & ws,
        std::shared_ptr<nfs_work_cofac> wc_p,
        std::shared_ptr<nfs_aux> aux_p,
        ALGO::special_q_data const & Q,
        thread_pool & pool,
        where_am_I & w,
        std::shared_ptr<std::vector<gpu_sieve_region>> regions,
        task_group * tg = nullptr);

#endif	/* CADO_LAS_PROCESS_BUCKET_REGION_HPP */
