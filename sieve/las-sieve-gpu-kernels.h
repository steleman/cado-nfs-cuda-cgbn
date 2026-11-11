#ifndef CADO_LAS_SIEVE_GPU_KERNELS_H
#define CADO_LAS_SIEVE_GPU_KERNELS_H

/* GPU siever (las-sieve-gpu.cpp): the sieve region of one special-q,
 * both sides, on the GPU. C interface.
 *
 * For each side: the lognorms (las-norms.cpp, lognorm_smart, same double
 * arithmetic), minus, saturating, the sum modulo 256 of the logs of the
 * factor base entries that hit each position (las-process-bucket-region
 * SminusS). The bucket-sieved entries are walked as p-lattices; the
 * small sieve entries are those that small_sieve_init made for this
 * special-q. Only positions with gcd(i,j) = 1 are meant to agree with
 * the CPU code: the others never survive.
 */

#include <stddef.h>
#include <stdint.h>

#include "las-fill-gpu-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* a bucket-sieved entry: lfg_lat plus its slice's logp. With
 * transformed != 0, r and proj are already in the q-lattice (powers of
 * 2, transformed on the host). */
struct lsg_lat {
    struct lfg_lat L;
    uint8_t logp;
    uint8_t transformed;
    uint16_t pad;
};

/* a small sieve entry of this special-q.
 *   kind 0: positions with i == r*j (mod q)
 *   kind 1: rows j == 0 (mod g), positions with i == r*(j/g) (mod q)
 *           (projective, ssp_t's q, g, U) */
struct lsg_small {
    uint32_t q, r, g;
    uint8_t logp;
    uint8_t kind;
    uint8_t resieve;
    uint8_t pad;
};

/* lognorm_smart of one side */
struct lsg_norm {
    int degree;             /* 1: u0*j + u1*i; > 1: the G segments */
    double u0, u1;
    double offset, modscale;
    uint8_t origin;         /* the value at (1,0) */
    uint32_t nseg;          /* G: nseg equations, nseg+1 endpoints */
    double cexp2[257];
};

/* the bucket-sieved entries of a factor base, on the GPU */
struct lsg_table;
struct lsg_table * lsg_table_create(const struct lsg_lat * lats, size_t n);
void lsg_table_free(struct lsg_table *);

struct lsg_side {
    struct lsg_norm norm;
    const double * G_endpoints;     /* host, nseg + 1 */
    const double * G_equations;     /* host, 2 * nseg */
    const struct lsg_table * table; /* the factor base's entries */
    const struct lsg_lat * lats;    /* host, nlats: more, for this q */
    size_t nlats;
    const struct lsg_small * small; /* host, nsmall */
    size_t nsmall;
};

struct lsg_qinfo {
    int64_t a0, b0, a1, b1;
    uint64_t qfac[8];
    int nqfac;
    int logI;
    uint32_t J;
};

/* survivors and what resieving finds on them */
struct lsg_survivor {
    uint32_t x;         /* position, row major */
    uint8_t S[2];       /* the sieve values */
    uint16_t pad;
};
struct lsg_hit_large {  /* a bucket-sieved entry that hits a survivor */
    uint32_t x;
    uint16_t slice;     /* as in lsg_lat */
    uint16_t hint;
    uint32_t side;
    uint32_t p;         /* the prime (for the cofactor filter) */
};
struct lsg_hit_small {  /* a resieved small sieve entry that hits */
    uint32_t x;
    uint32_t p;         /* the prime recorded (as resieve_small_bucket_region) */
    uint32_t side;
};
struct lsg_out {
    const struct lsg_survivor * surv; size_t nsurv;
    const struct lsg_hit_large * large; size_t nlarge;
    const struct lsg_hit_small * small; size_t nsmall;
    size_t nrejected;   /* survivors dropped by the cofactor filter */
};

/* The cofactor filter (las-sieve-gpu-cofilter.cu): for each survivor and
 * each side, the cofactor that las computes in cofactoring_sync() (the
 * norm |F(i,j)|, all its factors 2, the primes of the hits at that
 * position, the trial division primes and, on the special-q side if q is
 * composite, its prime factors removed) and check_leftover_norm() on it.
 * A survivor that check_leftover_norm() rejects on a side gets pad = 1;
 * whatever the GPU cannot decide (a cofactor of more than 126 bits that
 * passes the size tests, a zero norm) is left to the CPU. */
#define LSG_COF_MAXDEG 8
struct lsg_cofilter_side {
    int32_t deg;                /* < 0: this side is not checked */
    int8_t sign[LSG_COF_MAXDEG + 1];     /* of the coefficients of fij */
    uint32_t limbs[LSG_COF_MAXDEG + 1][16];  /* |coefficient|, 512 bits */
    uint32_t lpb, mfb;
    uint64_t lim;
    uint32_t nqf, qf[8];        /* q's factors (composite q, its side) */
    const uint32_t * td;        /* the trial division primes */
    uint32_t ntd;
};
struct lsg_cofilter {
    int bits;                   /* the CGBN width, 256 or 512; 0: off */
    uint64_t td_version;        /* changes when the td lists do */
    struct lsg_cofilter_side s[2];
};

struct lsg_ctx;

struct lsg_ctx * lsg_ctx_create(void);
/* the CUDA devices this process can see (0 if none) */
int lsg_device_count(void);
void lsg_ctx_free(struct lsg_ctx *);

/* Sieves both sides; the final sieve arrays (I*J bytes each, row major,
 * i from -I/2) are copied to S[side] if S != NULL and S[side] != NULL.
 * If out != NULL, also finds the survivors (S[side] <= bound[side] on
 * both sides, gcd(i,j) = 1, only (1,0) on row 0) and resieves them;
 * the arrays belong to the context and stay valid until its next call.
 * The small sieve entries with resieve != 0 are resieved. */
/* accept (with out): if not NULL, a survivor (s0, s1) is kept only if
 * bit s0 * 256 + s1 of accept (8 KB) is set: las's estimate filter
 * (filter_survivors_by_region), applied before resieving. accept_version
 * changes whenever its contents do (it is copied to the GPU then); the
 * array must stay valid until the call returns. */
void lsg_sieve(struct lsg_ctx *, const struct lsg_qinfo *,
        const struct lsg_side * sides, int nsides, uint8_t ** S,
        const uint8_t * bound, struct lsg_out * out,
        const uint32_t * accept, uint64_t accept_version,
        const struct lsg_cofilter * cof);

void lsg_ctx_stats(const struct lsg_ctx *, double * seconds, unsigned long * calls);

#ifdef __cplusplus
}
#endif

#endif /* CADO_LAS_SIEVE_GPU_KERNELS_H */
