#ifndef CADO_FACUL_GPU_HPP
#define CADO_FACUL_GPU_HPP

/* GPU cofactorization: the facul method chains of a strategy, run on
 * the GPU for many cofactors at once. Built when CUDA is available
 * (HAVE_CUDA); facul_gpu::get() returns nullptr otherwise, or when there
 * is no GPU, or with CADO_LAS_GPU_COFAC=0.
 *
 * A chain is the methods of one strategy entry (the vector that
 * facul_strategies::operator() returns) for one side. A job is one
 * cofactor (odd, below 2^126) with its chain; its result is what facul
 * would have found on that side: the status (smooth, not smooth, maybe)
 * and the primes. See facul-gpu-kernels.cu for how closely it follows
 * the CPU code.
 */

#include <cstdint>

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "cxx_mpz.hpp"
#include "facul-gpu-data.h"

struct facul_method_side;
struct facul_method;
struct facul_strategies;
struct fgpu_ctx;

/* What facul_all + factor_leftover_norms (las-cofactor.cpp) give for one
 * candidate. */
struct facul_gpu_outcome {
    bool cpu = false;   /* not handled: run the CPU code on it */
    int status = 0;     /* facul_status, as factor_leftover_norms */
    std::vector<std::vector<cxx_mpz>> primes;   /* per side, if smooth */
};

class facul_gpu {
  public:
    /* the process-wide engine, created on first use */
    static facul_gpu * get();

    /* chain id, or -1 if a method of this chain has no GPU version (the
     * caller then uses the CPU). Thread-safe. */
    int chain(std::vector<facul_method_side> const & methods, int side,
            unsigned long lpb, double BB, double BBB);

    /* Runs the jobs. A job whose cofactor is above 2^126 or even gets
     * FGPU_CPU. Thread-safe (one batch on the GPU at a time). */
    void run(std::vector<fgpu_job> const & jobs, std::vector<fgpu_result> & res);

    /* Cofactors a batch of candidates (all the norms of each) with the
     * strategies, as facul_all would, the GPU taking every side it can.
     * Candidates it cannot do get cpu == true. */
    void factor_all(std::vector<std::vector<cxx_mpz> const *> const & N,
            facul_strategies const & strat,
            std::vector<facul_gpu_outcome> & out);

    /* helpers */
    static bool fits(cxx_mpz const & n);
    static fgpu_job make_job(cxx_mpz const & n, int chain);
    static cxx_mpz prime(fgpu_result const & r, int i);

    int device() const;
    void stats(double & kernel_seconds, unsigned long & njobs) const;

    ~facul_gpu();
    facul_gpu(facul_gpu const &) = delete;
    facul_gpu & operator=(facul_gpu const &) = delete;

  private:
    explicit facul_gpu(fgpu_ctx * ctx);
    fgpu_ctx * ctx;

    std::mutex lock;        /* chain building */
    std::mutex run_lock;    /* the GPU */
    bool dirty = false;

    std::vector<fgpu_chain> chains;
    std::vector<fgpu_method> methods;
    std::vector<uint64_t> p64;
    std::vector<uint32_t> p32;
    std::vector<uint16_t> p16;
    std::vector<uint8_t> p8;

    /* per CPU method: its GPU record without the flags, or ok == false */
    struct method_entry { bool ok; fgpu_method m; };
    std::map<facul_method const *, method_entry> method_cache;
    std::map<std::pair<void const *, int>, int> chain_cache;

    method_entry const & method(facul_method const * fm);
    fgpu_bigint bigint(cxx_mpz const & z);
    void upload();
};

#endif /* CADO_FACUL_GPU_HPP */
