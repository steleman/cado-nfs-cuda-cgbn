#ifndef CADO_LAS_FILL_GPU_KERNELS_H
#define CADO_LAS_FILL_GPU_KERNELS_H

/* GPU top-level bucket fill (las-fill-gpu.cpp). C interface, so that the
 * C++ side needs no CUDA headers.
 *
 * A side's table lists the (entry, root) pairs of the top-level slices
 * of its factor base. For one special-q, lfg_fill transforms each root
 * into the q-lattice, reduces the p-lattice (Franke-Kleinjung) and
 * enumerates its points, as fill_in_buckets_toplevel does, and returns
 * the bucket updates grouped by slice, then bucket (the order in which a
 * bucket array stores them). The lattices that the CPU treats apart
 * (projective-like, vertical) are returned in a list instead.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lfg_lat {
    uint32_t q;         /* p^k, odd */
    uint32_t p;
    uint32_t r;         /* root, as fb_root_p1::r */
    uint16_t hint;      /* offset of the entry in its slice */
    uint16_t slice;     /* slice, numbered from 0 within the table */
    uint32_t proj;      /* 1 if the root is projective */
};

struct lfg_special {
    uint32_t q, r;      /* the transformed root */
    uint16_t hint, slice;
    uint32_t proj;
};

struct lfg_qinfo {
    int64_t a0, b0, a1, b1;     /* q-lattice basis (31 bits each) */
    uint64_t qfac[8];           /* prime factors of the special-q */
    int nqfac;
    int logI;
    uint32_t J;
    int logB;                   /* log2 of the bucket region size */
    uint32_t nbuckets;
    int hint_bytes;             /* 2 (shorthint_t) or 0 (emptyhint_t) */
};

struct lfg_table;
struct lfg_ctx;

/* NULL if there is no GPU */
struct lfg_ctx * lfg_ctx_create(void);
void lfg_ctx_free(struct lfg_ctx *);

struct lfg_table * lfg_table_create(const struct lfg_lat * lats, size_t n,
        uint32_t nslices);
void lfg_table_free(struct lfg_table *);

/* Runs the fill. On return, *counts points to nslices * nbuckets counts
 * (slice major), *updates to the updates in that order (4 bytes each
 * with a hint, 2 without), *specials to the special lattices. The
 * buffers belong to the context and stay valid until its next call. */
void lfg_fill(struct lfg_ctx *, const struct lfg_table *,
        const struct lfg_qinfo *,
        const uint32_t ** counts, const void ** updates, size_t * nupdates,
        const struct lfg_special ** specials, size_t * nspecials);

/* seconds spent in lfg_fill (GPU and copies), and calls */
void lfg_ctx_stats(const struct lfg_ctx *, double * seconds, unsigned long * calls);

#ifdef __cplusplus
}
#endif

#endif /* CADO_LAS_FILL_GPU_KERNELS_H */
