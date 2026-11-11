#ifndef CADO_LAS_COFAC_GPU_HPP
#define CADO_LAS_COFAC_GPU_HPP

/* Cofactorization of las survivors on the GPU (ecm/facul-gpu.hpp).
 *
 * process_bucket_region hands each candidate to submit() instead of
 * making a detached_cofac task of it. The candidates of all threads
 * (and subjobs) go into one batch; a thread of this class runs the batch
 * through the GPU and then queues, on the thread pool the candidate came
 * from, a task that does the rest of detached_cofac (statistics,
 * duplicate check, printing the relation) with the GPU's result.
 * Candidates the GPU cannot do get the CPU code in that task.
 *
 * A batch starts when it holds CADO_LAS_GPU_BATCH candidates (65536), or
 * when its oldest candidate has waited CADO_LAS_GPU_WAIT seconds (1), or
 * when flush() is called. Large batches matter: a launch lasts as long as
 * one candidate's whole chain of methods, whatever the batch size, until
 * the batch fills the GPU (some 20,000 candidates on an RTX PRO 4500).
 * flush(pool) returns once every candidate of that pool has its task
 * queued; las_subjob calls it before draining the pool.
 *
 * Off with CADO_LAS_GPU_COFAC=0, or when there is no GPU.
 */

#include <memory>

class thread_pool;
class nfs_work_cofac;
class nfs_aux;
struct cofac_standalone;

class las_cofac_gpu {
  public:
    /* nullptr when the GPU is not used */
    static las_cofac_gpu * get();

    void submit(thread_pool & pool,
            std::shared_ptr<nfs_work_cofac> const & wc,
            std::shared_ptr<nfs_aux> const & aux,
            cofac_standalone && cur);

    void flush(thread_pool & pool);

    /* one line for the end of the las output */
    static void report();

  private:
    struct impl;
    std::unique_ptr<impl> d;
    las_cofac_gpu();
    ~las_cofac_gpu();
};

#endif /* CADO_LAS_COFAC_GPU_HPP */
