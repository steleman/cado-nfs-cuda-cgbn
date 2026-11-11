#ifndef CADO_LAS_FILL_GPU_HPP
#define CADO_LAS_FILL_GPU_HPP

/* Top-level bucket fill on the GPU, see las-fill-gpu.cpp. */

class nfs_work;
class nfs_aux;
class thread_pool;
struct qlattice_basis;
struct where_am_I;

/* Queues the fill of this side's top-level buckets, computed on the GPU,
 * on the pool. Returns false (and does nothing) when it does not apply:
 * no GPU, CADO_LAS_GPU_FILL=0, top level above 1, sublattices. */
bool fill_in_buckets_toplevel_gpu(nfs_work & ws, nfs_aux & aux,
        qlattice_basis const & Q, thread_pool & pool, int side,
        where_am_I & w);

/* one line for the end of the las output */
void las_fill_gpu_report();

#endif /* CADO_LAS_FILL_GPU_HPP */
