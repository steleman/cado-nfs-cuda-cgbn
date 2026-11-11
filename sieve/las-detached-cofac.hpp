#ifndef CADO_LAS_DETACHED_COFAC_HPP
#define CADO_LAS_DETACHED_COFAC_HPP

#include <memory>

#include "las-cofac-standalone.hpp"

/* gpu: the result of the GPU cofactorization (las-cofac-gpu.hpp) for
 * this candidate, if it has one */
class nfs_aux;
class nfs_work_cofac;
class worker_thread;
struct relation;
struct facul_gpu_outcome;

extern relation detached_cofac(worker_thread * worker,
        nfs_work_cofac & wc,
        nfs_aux & aux,
        cofac_standalone && cur,
        facul_gpu_outcome const * gpu = nullptr);

#endif	/* CADO_LAS_DETACHED_COFAC_HPP */
