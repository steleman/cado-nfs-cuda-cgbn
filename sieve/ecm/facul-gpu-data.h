#ifndef CADO_FACUL_GPU_DATA_H
#define CADO_FACUL_GPU_DATA_H

/* Data shared by the host side (facul-gpu.cpp) and the kernels
 * (facul-gpu-kernels.cu) of the GPU cofactorization: the description of
 * the method chains, and the jobs and their results.
 *
 * A chain is the list of methods that facul applies to one side of a
 * candidate (the methods of a strategy whose side is that side, in
 * order). All of it lives in one blob that is copied to the GPU as is:
 *
 *   fgpu_blob_header
 *   fgpu_chain[nchains]
 *   fgpu_method[nmethods]
 *   pool64[n64]   multi-word integers (P-1 exponents, curve constants)
 *   pool32[n32]   stage 2 U lists
 *   pool16[n16]   stage 2 pair lists
 *   pool8[n8]     PRAC bytecode
 *
 * Offsets in the method records are indices into the pools.
 */

#include <stdint.h>

#define FGPU_PM1 1
#define FGPU_PP1 2
#define FGPU_ECM 3

#define FGPU_FLAG_STOP_UNSPLIT 1u
#define FGPU_FLAG_LAST 2u

/* stage 2 pair list markers (uint16) */
#define FGPU_PAIR_INCR_V 0xfffeu
#define FGPU_PAIR_END 0xffffu

/* largest U list (w < 1000 in stage2.cpp, so phi(w)/2 <= 120) */
#define FGPU_UMAX 120

/* most primes and pending composites one side may have */
#define FGPU_MAXP 8
#define FGPU_MAXC 4

/* result status, as facul_status, plus "redo this one on the CPU" */
#define FGPU_NOT_SMOOTH (-1)
#define FGPU_MAYBE 0
#define FGPU_SMOOTH 1
#define FGPU_CPU 2

struct fgpu_bigint {
    uint32_t off;      /* in pool64, little endian limbs */
    uint32_t len;      /* number of limbs (0 means 0) */
    int32_t sign;      /* -1, 0 or 1 */
    uint32_t pad;
};

struct fgpu_method {
    uint32_t type;      /* FGPU_PM1 / FGPU_PP1 / FGPU_ECM */
    uint32_t flags;     /* FGPU_FLAG_* */
    uint32_t B1, B2, exp2;
    uint32_t bc_off, bc_len;    /* PRAC bytecode, in pool8 */
    uint32_t E_off, E_len;      /* P-1 stage 1 exponent, in pool64 */
    /* stage 2 plan (stage2.h), w == 0 means no stage 2 */
    uint32_t w, U_len, vmin, vmax;
    uint32_t U_off;             /* pool32 */
    uint32_t pairs_off;         /* pool16 */
    uint32_t pad;
    /* P+1: start value c[0]/c[1].
     * ECM: b = (A+2)/4 = c[0]/c[1], starting point (c[2] :: c[3]) */
    struct fgpu_bigint c[4];
};

struct fgpu_chain {
    uint32_t m0, nm;    /* methods m0 .. m0+nm-1 */
    uint32_t lpb;
    uint32_t pad;
    double BB, BBB;     /* B^2, B^3 */
};

struct fgpu_blob_header {
    uint32_t nchains, nmethods;
    uint32_t off_chains, off_methods;   /* bytes from the blob start */
    uint32_t off64, off32, off16, off8;
    uint32_t n64, n32, n16, n8;
};

struct fgpu_job {
    uint64_t n[2];      /* the cofactor, odd, < 2^126 */
    uint32_t chain;
    uint32_t pad;
};

struct fgpu_result {
    int32_t status;     /* FGPU_* */
    int32_t nprimes;
    uint64_t p[FGPU_MAXP][2];
};

#endif /* CADO_FACUL_GPU_DATA_H */
