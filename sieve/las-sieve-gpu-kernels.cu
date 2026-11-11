/* GPU siever, see las-sieve-gpu-kernels.h.
 *
 * Compile without FMA contraction (-fmad=false): the lognorms must be
 * computed with the same double operations as las-norms.cpp, which GCC
 * compiles without fused multiply-adds (no -mfma).
 *
 * Kernels, per side:
 *   lsg_norms   one warp per row: lognorm_smart::fill_rat_inner /
 *               fill_alg for that row (row 0: only (1,0) is not 255)
 *   lsg_large   one thread per bucket-sieved (entry, root): transform,
 *               reduce, walk the p-lattice over the whole region, adding
 *               logp to 16-bit sums
 *   lsg_small   one block per tile of the region, a warp per (entry,
 *               row): the small sieve entries into 16-bit sums in shared
 *               memory, then
 *               S = max(norm - ((large + small) mod 256), 0)
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
#include <vector>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"

#include "las-sieve-gpu-kernels.h"
#include "las-sieve-gpu-cofilter.h"
#include "las-gpu-lattice.cuh"

#define CK(x) do {                                                      \
    cudaError_t const e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                            \
        fprintf(stderr, "las-sieve-gpu: %s failed: %s\n", #x,           \
                cudaGetErrorString(e_));                                \
        abort();                                                        \
    }                                                                   \
} while (0)

#define LOGNORM_GUARD_BITS 1
#define TILE_LOG 14     /* positions per small sieve tile (32 KB of
                           shared sums: 3 blocks an SM; 2^15 allowed 1) */
/* positions sieved at a time. The arrays of a slab (norms, sums and S of
 * both sides, 16 bytes a position) of the dozen subjobs sharing a GPU
 * compete for its L2 cache (64 MB on an RTX PRO 4500): 2^22 positions,
 * 64 MB per subjob, against 2^25 before. 16 subjobs: 4% faster at I=12
 * (one slab -> 2), 2.1x at I=14, 2.0x at I=15; 2^21 and 2^23 slower. */
#define SLAB_LOG 22     /* for a 64 MB L2; see slab_log() */
#define SLAB_LOG_MIN 17         /* whole rows (logI <= 16), whole tiles */
#define SLAB_LOG_MAX 26
static_assert(SLAB_LOG_MIN > 16 && SLAB_LOG_MIN >= TILE_LOG, "slabs of whole rows, whole tiles");

/* {{{ lognorms: las-norms.cpp, line by line */
__device__ __forceinline__ uint8_t compute_y(double G, double offset, double modscale)
{
    double const v = G + 1.;
    /* lg2_raw: the top 32 bits of the double, as a number */
    double const raw = (double) ((unsigned long long) __double_as_longlong(v) >> 32);
    double const y = (raw - offset) * modscale;
    /* double -> unsigned char as x86 GCC does it: through int */
    return (uint8_t) (int) y;
}

/* The lognorm algorithm is sequential along a row: one thread per row
 * (it was a warp per row, all 32 lanes computing the same values, which
 * kept the FP64 pipe 72% busy for nothing). Each thread writes its row
 * through a 16-byte buffer, so that the stores are 16 bytes wide. */
struct row_writer {
    uint8_t * base;     /* the row, 16-byte aligned */
    uint32_t pos;       /* next position in the row */
    uint32_t buf[4];
    __device__ __forceinline__ void flush()
    {
        *(uint4 *) (base + (pos & ~15u)) = make_uint4(buf[0], buf[1], buf[2], buf[3]);
    }
    __device__ __forceinline__ void put(uint8_t y)
    {
        uint32_t const k = pos & 15;
        uint32_t & w = buf[k >> 2];
        uint32_t const sh = 8 * (k & 3);
        w = (w & ~(0xffu << sh)) | ((uint32_t) y << sh);
        if (k == 15) flush();
        pos++;
    }
    __device__ __forceinline__ void fill(uint8_t y, size_t n)
    {
        for ( ; n && (pos & 15) ; n--) put(y);
        uint32_t const r = 0x01010101u * y;
        for ( ; n >= 16 ; n -= 16, pos += 16)
            *(uint4 *) (base + pos) = make_uint4(r, r, r, r);
        for ( ; n ; n--) put(y);
    }
};

/* A row_writer for a segment of a row that other threads write the rest
 * of: whole 16-byte chunks inside [lo, ...) are stored at once, the
 * partial ones at the segment's ends byte by byte, so that the segments
 * of a row never overwrite each other. */
struct seg_writer {
    uint8_t * base;     /* the row, 16-byte aligned */
    uint32_t pos;       /* next position in the row */
    uint32_t lo;        /* the segment's first position */
    uint32_t buf[4];
    __device__ __forceinline__ void store_bytes(uint32_t from, uint32_t to)
    {
        for (uint32_t k = from; k < to; k++)
            base[k] = (uint8_t) (buf[(k & 15) >> 2] >> (8 * (k & 3)));
    }
    __device__ __forceinline__ void flush()
    {
        uint32_t const c = pos & ~15u;
        if (c >= lo)
            *(uint4 *) (base + c) = make_uint4(buf[0], buf[1], buf[2], buf[3]);
        else
            store_bytes(lo, c + 16);
    }
    __device__ __forceinline__ void put(uint8_t y)
    {
        uint32_t const k = pos & 15;
        uint32_t & w = buf[k >> 2];
        uint32_t const sh = 8 * (k & 3);
        w = (w & ~(0xffu << sh)) | ((uint32_t) y << sh);
        if (k == 15) flush();
        pos++;
    }
    __device__ __forceinline__ void fill(uint8_t y, size_t n)
    {
        for ( ; n && (pos & 15) ; n--) put(y);
        uint32_t const r = 0x01010101u * y;
        for ( ; n >= 16 ; n -= 16, pos += 16)
            *(uint4 *) (base + pos) = make_uint4(r, r, r, r);
        for ( ; n ; n--) put(y);
    }
    /* the partial chunk at the end */
    __device__ __forceinline__ void finish()
    {
        uint32_t const c = pos & ~15u;
        if (c != pos) store_bytes(c > lo ? c : lo, pos);
    }
};

/* fill_rat_inner for one row j, positions i0 .. i1 (the writer is at i0) */
template<typename W_t>
__device__ void fill_rat_row(W_t & W, int i0, int i1, unsigned int j,
        double u0, double u1, double c, lsg_norm const & N)
{
    double const offset = N.offset, modscale = N.modscale;
    if (u1 == 0) {
        double const g = fabs(u0 * j + c);
        uint8_t const y = compute_y(g, offset, modscale);
        W.fill(y, i1 - i0);
        return;
    }
    double const u1_abs = fabs(u1);
    double const inv_u1_abs = 1 / u1_abs;

    int i = i0;
    double const t0 = u0 * j + c;
    double g = fabs(t0 + u1 * i0);
    uint8_t y;
    double const root = -t0 / u1;
    bool root_ahead = false;

    if (g * (double) (1ULL << 51) >= fabs(u0 * j) + fabs(c)) {
        y = compute_y(g, offset, modscale);
        root_ahead = root >= i0;
    } else {
        y = LOGNORM_GUARD_BITS;
    }

    if (root_ahead) {
        for (; i < i1; y--) {
            double ix = root - N.cexp2[y] * inv_u1_abs;
            ix += (ix >= 0);
            if (ix >= (double) i1)
                ix = i1;
            size_t const di = (size_t) ((int) ix - i);
            i = (int) ix;
            if (!di)
                break;
            W.fill(y, di);
        }
        if (i >= i1)
            return;
        g = t0 + u1 * i;
        for (; i < i1 && signbit(g) != signbit(u1); i++) {
            W.put(compute_y(fabs(g), offset, modscale));
            g += u1;
        }
        if (i >= i1)
            return;
        g = fabs(g);
        y = compute_y(g, offset, modscale);
    }

    for (; i < i1;) {
        W.put(y);
        i++;
        g += u1_abs;
        uint8_t const oy = y;
        y = compute_y(g, offset, modscale);
        if (y == oy)
            break;
    }
    if (i >= i1)
        return;

    for (; i < i1; y++) {
        double ix = root + N.cexp2[(unsigned int) y + 1] * inv_u1_abs;
        ix += (ix >= 0);
        if (ix >= (double) i) {
            if (ix >= i1)
                ix = i1;
            size_t const di = (size_t) ((int) ix - i);
            i = (int) ix;
            W.fill(y, di);
        }
    }
}

__global__ void lsg_norms(uint8_t * S, lsg_norm N, const double * G_end,
        const double * G_eq, int logI, uint32_t j_lo, uint32_t nrows)
{
    /* one thread per row; S starts at row j_lo */
    uint32_t const drow = blockIdx.x * blockDim.x + threadIdx.x;
    if (drow >= nrows) return;
    uint32_t const j = j_lo + drow;
    int const I = 1 << logI;
    int const i0 = -I / 2, i1 = I / 2;
    row_writer W;
    W.base = S + ((size_t) drow << logI);
    W.pos = 0;
    W.buf[0] = W.buf[1] = W.buf[2] = W.buf[3] = 0;
    if (j == 0) {
        W.fill(255, 1 - i0);
        W.put(N.origin);
        W.fill(255, I - (1 - i0) - 1);
        return;
    }
    if (N.degree <= 1) {
        double const c = (N.u0 * 0.0 + N.u1 * 0.0) / 1;
        fill_rat_row(W, i0, i1, j, N.u0, N.u1, c, N);
        return;
    }
    /* fill_alg */
    double const dj = 0.0, di = 0.0;
    int Gi0 = i0;
    double const jr = j + dj;
    double mj1 = jr;
    for (int d = N.degree; --d > 1;)
        mj1 *= jr;
    for (uint32_t k = 0; k < N.nseg; k++) {
        double const r1 = G_end[k + 1];
        double const e0 = G_eq[2 * k], e1 = G_eq[2 * k + 1];
        double const x1 = jr * r1 - di;
        int Gi1 = (int) (x1 + (x1 >= 0));
        if (Gi0 >= i1) {
            break;
        } else if (Gi1 > Gi0) {
            Gi1 = min(Gi1, i1);
            fill_rat_row(W, Gi0, Gi1, j, e0 * mj1, e1 * mj1,
                    (e0 * dj + e1 * di) * mj1, N);
            Gi0 = Gi1;
        }
    }
}
/* The algebraic side (degree > 1): its rows are made of segments (the G
 * equations) that fill_rat_row does each from scratch, so a thread per
 * (segment, row) gives the bytes of lsg_norms, with chains several times
 * shorter (lsg_norms was the largest kernel: c140, I=13, 551 us a launch
 * on this side against 276 on the rational one). Thread t is segment
 * t / nrows of row t % nrows: a warp has the same segment of 32 rows. */
__global__ void lsg_norms_alg(uint8_t * S, lsg_norm N, const double * G_end,
        const double * G_eq, int logI, uint32_t j_lo, uint32_t nrows)
{
    uint64_t const t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const k = (uint32_t) (t / nrows), drow = (uint32_t) (t % nrows);
    if (k >= N.nseg) return;
    uint32_t const j = j_lo + drow;
    int const I = 1 << logI;
    int const i0 = -I / 2, i1 = I / 2;
    uint8_t * const row = S + ((size_t) drow << logI);
    if (j == 0) {
        if (k) return;
        row_writer W;
        W.base = row; W.pos = 0;
        W.buf[0] = W.buf[1] = W.buf[2] = W.buf[3] = 0;
        W.fill(255, 1 - i0);
        W.put(N.origin);
        W.fill(255, I - (1 - i0) - 1);
        return;
    }
    double const dj = 0.0, di = 0.0;
    double const jr = j + dj;
    double mj1 = jr;
    for (int d = N.degree; --d > 1;)
        mj1 *= jr;
    /* where segment k starts: the loop of lsg_norms, without the fills */
    int Gi0 = i0;
    for (uint32_t k2 = 0; k2 < k; k2++) {
        double const x1 = jr * G_end[k2 + 1] - di;
        int const Gi1 = (int) (x1 + (x1 >= 0));
        if (Gi0 >= i1) return;
        if (Gi1 > Gi0) Gi0 = min(Gi1, i1);
    }
    if (Gi0 >= i1) return;
    double const r1 = G_end[k + 1];
    double const e0 = G_eq[2 * k], e1 = G_eq[2 * k + 1];
    double const x1 = jr * r1 - di;
    int Gi1 = (int) (x1 + (x1 >= 0));
    if (Gi1 <= Gi0) return;
    Gi1 = min(Gi1, i1);
    seg_writer W;
    W.base = row;
    W.pos = W.lo = (uint32_t) (Gi0 - i0);
    W.buf[0] = W.buf[1] = W.buf[2] = W.buf[3] = 0;
    fill_rat_row(W, Gi0, Gi1, j, e0 * mj1, e1 * mj1,
            (e0 * dj + e1 * di) * mj1, N);
    W.finish();
}
/* }}} */

/* {{{ large primes */
/* setup_lattice of las-gpu-lattice.cuh, but every lattice is walked
 * (the CPU's special cases enumerate the same points) */
__device__ bool setup_walk(lsg_lat const & E, lfg_qinfo const & Q, lfg_state & st)
{
    lfg_lat const & lat = E.L;
    for (int k = 0; k < Q.nqfac; k++)
        if (Q.qfac[k] == lat.p) return false;
    uint32_t const p = lat.q;
    uint32_t r;
    bool proj;
    if (E.transformed) {
        r = lat.r;
        proj = lat.proj;
    } else {
        int64_t aux1, aux2;
        if (!lat.proj) {
            aux1 = (int64_t) lat.r * Q.b1 - Q.a1;
            aux2 = Q.a0 - (int64_t) lat.r * Q.b0;
        } else {
            aux1 = Q.b1 - (int64_t) lat.r * Q.a1;
            aux2 = (int64_t) lat.r * Q.a0 - Q.b0;
        }
        uint32_t const u = mod_signed(aux1, p);
        uint32_t const v = mod_signed(aux2, p);
        uint32_t const iv = inv_mod(v, p);
        if (iv) {
            r = (uint32_t) (((uint64_t) u * iv) % p);
            proj = false;
        } else {
            uint32_t const iu = inv_mod(u, p);
            if (!iu) return false;
            r = (uint32_t) (((uint64_t) v * iu) % p);
            proj = true;
        }
    }
    uint32_t const I = 1u << Q.logI;
    pli_t L;
    if (!proj) {
        L.mi0 = p; L.j0 = 0; L.i1 = r; L.j1 = 1;
    } else {
        unsigned long t;
        unsigned long const g = xgcd_ul(&t, r, p);
        L.mi0 = (uint32_t) (p / g); L.j0 = 0; L.i1 = (uint32_t) t; L.j1 = (uint32_t) g;
    }
    reduce(L, I);
    int const logI = Q.logI;
    st.inc_warp = ((uint64_t) L.j0 << logI) + (int64_t) (-(int32_t) L.mi0);
    st.inc_step = (L.i1 >> logI) ? UINT64_MAX
                : ((uint64_t) L.j1 << logI) + (int64_t) L.i1;
    st.bound_step = L.mi0;
    st.bound_warp = (L.i1 >> logI) ? 0 : I - L.i1;
    st.x = (uint64_t) 1 << (logI - 1);
    {
        uint32_t const i = st.x & (I - 1);
        if (i >= st.bound_warp) st.x += st.inc_warp;
        if (i < st.bound_step) st.x += st.inc_step;
    }
    if (L.mi0 == 0 && L.j0 == 0) return false;
    return true;
}

/* The walk of each p-lattice over the slab [lo, hi) of the region. Its
 * state is split: the steps (lsg_cst, fixed for the special-q, written
 * once by setup_walk in the first slab) and the position x, in two
 * arrays, x of the slab and x of the next one. Only x changes from slab
 * to slab, and a lattice with no point in a slab (its x is past hi) costs
 * reading x alone: 16 bytes instead of the 136 that reading and writing
 * the whole 32-byte state cost in the sieving and resieving passes (c140:
 * 1.6 million lattices, 4 slabs a special-q). lo is even. */
struct lsg_cst {
    uint64_t inc_warp, inc_step;
    uint32_t bound_step;
    uint32_t bound_warp;    /* <= I <= 2^16; bits 24..31: logp (lsg_large
                               then need not read the lattice) */
};

__device__ __forceinline__ void load_walk(lfg_state & st, uint64_t x, lsg_cst const & C)
{
    st.x = x;
    st.inc_warp = C.inc_warp; st.inc_step = C.inc_step;
    st.bound_step = C.bound_step; st.bound_warp = C.bound_warp & 0xffffffu;
}

/* first: compute the walk (cst, and x: the start); else x from x_cur.
 * x_next (no resieving pass after this one): where the slab ends. */
__global__ void lsg_large(const lsg_lat * lats, uint32_t n, lfg_qinfo Q,
        bool first, uint64_t * x_cur, lsg_cst * cst, uint64_t * x_next,
        uint64_t lo, uint64_t hi, uint32_t * SS)
{
    uint32_t const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    lfg_state st;
    uint32_t add;
    if (first) {
        lsg_lat const E = lats[t];
        if (!setup_walk(E, Q, st)) st.x = UINT64_MAX;
        lsg_cst C;
        C.inc_warp = st.inc_warp; C.inc_step = st.inc_step;
        C.bound_step = st.bound_step;
        C.bound_warp = st.bound_warp | ((uint32_t) E.logp << 24);
        cst[t] = C;
        x_cur[t] = st.x;
        add = E.logp;
    } else {
        uint64_t const x = x_cur[t];
        if (x >= hi) {
            if (x_next) x_next[t] = x;
            return;
        }
        lsg_cst const C = cst[t];
        load_walk(st, x, C);
        add = C.bound_warp >> 24;
    }
    uint32_t const maskI = (1u << Q.logI) - 1;
    uint64_t const even_mask = ((uint64_t) 1 << Q.logI) | 1;
    for (; st.x < hi; step(st, maskI))
        if (st.x & even_mask) {
            uint64_t const y = st.x - lo;
            atomicAdd(&SS[y >> 1], add << (16 * (y & 1)));
        }
    if (x_next) x_next[t] = st.x;
}
/* }}} */

/* {{{ small primes, and the final values */
__global__ void lsg_small_tiles(const lsg_small * E, uint32_t n,
        uint32_t nwarp_entries, int logI, uint64_t lo,
        const uint32_t * SSl, const uint8_t * norms, uint8_t * S)
{
    extern __shared__ uint32_t sh[];     /* 2^TILE_LOG 16-bit sums */
    uint32_t const npos = 1u << TILE_LOG;
    /* positions relative to the slab; base is the absolute one */
    uint64_t const rel = (uint64_t) blockIdx.x << TILE_LOG;
    uint64_t const base = lo + rel;
    for (uint32_t k = threadIdx.x; k < npos / 2; k += blockDim.x) sh[k] = 0;
    __syncthreads();

    int const I = 1 << logI;
    /* rows of the tile, and the part of each row that the tile covers */
    uint32_t const j0 = (uint32_t) (base >> logI);
    uint32_t const nrows = logI >= TILE_LOG ? 1 : 1u << (TILE_LOG - logI);
    uint32_t const ilo = logI >= TILE_LOG ? (uint32_t) (base & (I - 1)) : 0;
    uint32_t const rowlen = logI >= TILE_LOG ? npos : (uint32_t) I;

    /* a warp per (entry, row), the lanes along the row */
    unsigned const lane = threadIdx.x & 31;
    uint32_t const warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    uint32_t const hI = (uint32_t) (I / 2);
    uint32_t const npairs = n * nrows;
    /* entries are sorted by q (host): the first nwarp_entries have many
     * hits per row and get a warp per (entry, row); the others a lane */
    uint32_t const nw = nwarp_entries * nrows;
    for (uint32_t pr = warp; pr < nw; pr += nwarps) {
        uint32_t const e = pr / nrows, dr = pr - e * nrows;
        lsg_small const s = E[e];
        uint32_t const q = s.q;
        uint32_t const j = j0 + dr;
        uint32_t jj = j;
        if (s.kind != 0) {
            if (j % s.g) continue;
            jj = j / s.g;
        }
        uint64_t const v = (uint64_t) s.r * jj + hI;
        uint32_t start = v >> 32 ? (uint32_t) (v % q) : (uint32_t) v % q;
        if (start < ilo) start += ((ilo - start + q - 1) / q) * q;
        uint32_t const iend = ilo + rowlen;
        uint32_t const add = (uint32_t) s.logp;
        for (uint32_t i = start + lane * q; i < iend; i += 32 * q) {
            /* not (0,0): every entry hits it, and past 65535 its 16-bit
             * sum carried into its neighbour (1,0) in the same word (I >=
             * 14). It is never a survivor; the CPU's value there is not
             * used either. */
            if (j == 0 && i == hI) continue;
            uint32_t const x = dr * rowlen + (i - ilo);
            atomicAdd(&sh[x >> 1], add << (16 * (x & 1)));
        }
    }
    for (uint32_t pr = nw + threadIdx.x; pr < npairs; pr += blockDim.x) {
        uint32_t const e = pr / nrows, dr = pr - e * nrows;
        lsg_small const s = E[e];
        uint32_t const q = s.q;
        uint32_t const j = j0 + dr;
        uint32_t jj = j;
        if (s.kind != 0) {
            if (j % s.g) continue;
            jj = j / s.g;
        }
        uint64_t const v = (uint64_t) s.r * jj + hI;
        uint32_t start = v >> 32 ? (uint32_t) (v % q) : (uint32_t) v % q;
        if (start < ilo) start += ((ilo - start + q - 1) / q) * q;
        uint32_t const iend = ilo + rowlen;
        uint32_t const add = (uint32_t) s.logp;
        for (uint32_t i = start; i < iend; i += q) {
            if (j == 0 && i == hI) continue;     /* (0,0): see above */
            uint32_t const x = dr * rowlen + (i - ilo);
            atomicAdd(&sh[x >> 1], add << (16 * (x & 1)));
        }
    }
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < npos; k += blockDim.x) {
        uint64_t const x = rel + k;
        uint32_t const sl = (SSl[x >> 1] >> (16 * (x & 1))) & 0xffff;
        uint32_t const ss = (sh[k >> 1] >> (16 * (k & 1))) & 0xffff;
        uint8_t const sum = (uint8_t) (sl + ss);
        uint8_t const nv = norms[x];
        S[x] = nv > sum ? nv - sum : 0;
    }
}
/* }}} */

/* {{{ survivors and resieving */
__device__ __forceinline__ bool coprime_ij(uint64_t x, int logI)
{
    long const i = (long) (x & ((1u << logI) - 1)) - (1L << (logI - 1));
    unsigned long const j = x >> logI;
    if (j == 0) return i == 1;
    unsigned long a = i < 0 ? -i : i, b = j;
    if (a == 0) return b == 1;
    /* binary gcd */
    int const k = __ffsll((long long) (a | b)) - 1;
    if (k) return false;
    a >>= __ffsll((long long) a) - 1;
    do {
        b >>= __ffsll((long long) b) - 1;
        if (a > b) { unsigned long const t = a; a = b; b = t; }
        b -= a;
    } while (b);
    return a == 1;
}

/* The bitmap of the survivors, and a summary of it: one bit per granule
 * of 2^glog positions, at most SUMMARY_WORDS words, which fits in shared
 * memory. */
#define SUMMARY_WORDS 4096
/* a position whose values are within the bounds */
__device__ __forceinline__ void lsg_survivor_at(uint64_t y, uint8_t s0, uint8_t s1,
        uint64_t lo, int logI, lsg_survivor * surv, uint32_t * nsurv,
        uint32_t cap, uint32_t * bitmap, uint32_t * summary, int glog,
        const uint32_t * accept)
{
    if (accept) {
        uint32_t const k = (uint32_t) s0 * 256 + s1;
        if (!((__ldg(accept + (k >> 5)) >> (k & 31)) & 1)) return;
    }
    uint64_t const x = lo + y;
    if (!coprime_ij(x, logI)) return;
    uint32_t const k = atomicAdd(nsurv, 1u);
    if (k < cap) {
        lsg_survivor v;
        v.x = (uint32_t) x; v.S[0] = s0; v.S[1] = s1; v.pad = 0;
        surv[k] = v;
    }
    atomicOr(&bitmap[y >> 5], 1u << (y & 31));
    uint64_t const g = y >> glog;
    atomicOr(&summary[g >> 5], 1u << (g & 31));
}

/* A thread per 16 positions (npos is a multiple of 16), read 16 bytes at
 * a time and compared 4 at a time: one thread per position, reading two
 * bytes each, made this kernel the largest by threads times time (41% at
 * c140) for about a thousand survivors a slab. y is relative to the
 * slab, x absolute; the bitmaps are per slab. */
__global__ void lsg_survivors(const uint8_t * S0, const uint8_t * S1,
        uint64_t lo, uint64_t npos, int logI, uint8_t b0, uint8_t b1,
        lsg_survivor * surv, uint32_t * nsurv, uint32_t cap, uint32_t * bitmap,
        uint32_t * summary, int glog, const uint32_t * accept)
{
    uint64_t const c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t const y0 = c * 16;
    if (y0 >= npos) return;
    uint4 const a = *(const uint4 *) (S0 + y0);
    uint4 const b = *(const uint4 *) (S1 + y0);
    uint32_t const B0 = 0x01010101u * b0, B1 = 0x01010101u * b1;
    uint32_t const m[4] = {
        __vcmpleu4(a.x, B0) & __vcmpleu4(b.x, B1),
        __vcmpleu4(a.y, B0) & __vcmpleu4(b.y, B1),
        __vcmpleu4(a.z, B0) & __vcmpleu4(b.z, B1),
        __vcmpleu4(a.w, B0) & __vcmpleu4(b.w, B1) };
    if (!(m[0] | m[1] | m[2] | m[3])) return;
    uint32_t const wa[4] = { a.x, a.y, a.z, a.w }, wb[4] = { b.x, b.y, b.z, b.w };
    for (int w = 0; w < 4; w++)
        for (int k = 0; k < 4; k++)
            if ((m[w] >> (8 * k)) & 0xff)
                lsg_survivor_at(y0 + 4 * w + k, (uint8_t) (wa[w] >> (8 * k)),
                        (uint8_t) (wb[w] >> (8 * k)), lo, logI, surv, nsurv, cap,
                        bitmap, summary, glog, accept);
}

/* from x_cur (which a redo of the slab reads again), to x_next */
__global__ void lsg_resieve_large(const lsg_lat * lats, uint32_t n, lfg_qinfo Q,
        const uint64_t * x_cur, const lsg_cst * cst, uint64_t * x_next,
        uint64_t lo, uint64_t hi,
        const uint32_t * bitmap, const uint32_t * summary, int glog,
        uint32_t nsummary, uint32_t side, lsg_hit_large * hits,
        uint32_t * nhits, uint32_t cap)
{
    /* a few blocks an SM, each over many lattices: copying the 16 KB
     * summary into every block of 128 lattices was the main cost (most
     * lattices have only a few points) */
    __shared__ uint32_t sum_sh[SUMMARY_WORDS];
    for (uint32_t k = threadIdx.x; k < nsummary; k += blockDim.x) sum_sh[k] = summary[k];
    __syncthreads();
    uint32_t const maskI = (1u << Q.logI) - 1;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n;
            t += gridDim.x * blockDim.x) {
    uint64_t const x = x_cur[t];
    if (x >= hi) {
        x_next[t] = x;
        continue;
    }
    lfg_state st;
    load_walk(st, x, cst[t]);
    for (; st.x < hi; step(st, maskI)) {
        uint64_t const y = st.x - lo;
        uint64_t const g = y >> glog;
        if (!((sum_sh[g >> 5] >> (g & 31)) & 1)) continue;
        if (!((__ldg(bitmap + (y >> 5)) >> (y & 31)) & 1)) continue;
        uint32_t const k = atomicAdd(nhits, 1u);
        if (k < cap) {
            lfg_lat const L = lats[t].L;
            lsg_hit_large h;
            h.x = (uint32_t) st.x; h.slice = L.slice; h.hint = L.hint; h.side = side;
            h.p = L.p;
            hits[k] = h;
        }
    }
    x_next[t] = st.x;
    }
}

/* lsg_resieve_large on the slab [lo, hi), then lsg_large on the next one
 * [hi, hi2) (its sums SS2, cleared): at c140 a lattice has about three
 * points a slab, so both passes were mostly the work per lattice (its
 * position and steps); this pays it once a slab instead of twice. x_next
 * gets the position at hi: the start of the next slab, which the next
 * pass resieves from. x_cur is not written, so the pass can be redone
 * (after an overflow, with SS2 cleared again). */
__global__ void lsg_resieve_sieve(const lsg_lat * lats, uint32_t n, lfg_qinfo Q,
        const uint64_t * x_cur, const lsg_cst * cst, uint64_t * x_next,
        uint64_t lo, uint64_t hi, uint64_t hi2,
        const uint32_t * bitmap, const uint32_t * summary, int glog,
        uint32_t nsummary, uint32_t side, lsg_hit_large * hits,
        uint32_t * nhits, uint32_t cap, uint32_t * SS2)
{
    __shared__ uint32_t sum_sh[SUMMARY_WORDS];
    for (uint32_t k = threadIdx.x; k < nsummary; k += blockDim.x) sum_sh[k] = summary[k];
    __syncthreads();
    uint32_t const maskI = (1u << Q.logI) - 1;
    uint64_t const even_mask = ((uint64_t) 1 << Q.logI) | 1;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n;
            t += gridDim.x * blockDim.x) {
        uint64_t const x = x_cur[t];
        if (x >= hi2) {
            x_next[t] = x;
            continue;
        }
        lsg_cst const C = cst[t];
        lfg_state st;
        load_walk(st, x, C);
        for (; st.x < hi; step(st, maskI)) {
            uint64_t const y = st.x - lo;
            uint64_t const g = y >> glog;
            if (!((sum_sh[g >> 5] >> (g & 31)) & 1)) continue;
            if (!((__ldg(bitmap + (y >> 5)) >> (y & 31)) & 1)) continue;
            uint32_t const k = atomicAdd(nhits, 1u);
            if (k < cap) {
                lfg_lat const L = lats[t].L;
                lsg_hit_large h;
                h.x = (uint32_t) st.x; h.slice = L.slice; h.hint = L.hint; h.side = side;
                h.p = L.p;
                hits[k] = h;
            }
        }
        x_next[t] = st.x;
        uint32_t const add = C.bound_warp >> 24;
        for (; st.x < hi2; step(st, maskI))
            if (st.x & even_mask) {
                uint64_t const y = st.x - hi;
                atomicAdd(&SS2[y >> 1], add << (16 * (y & 1)));
            }
    }
}

/* the primes of the hits, keyed by position and side, for the cofactor
 * filter: (x << 33) | (side << 32) | p */
__global__ void lsg_hit_keys(const lsg_hit_large * hl, uint32_t nl,
        const lsg_hit_small * hs, uint32_t ns, uint64_t * keys)
{
    uint32_t const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nl) {
        lsg_hit_large const h = hl[k];
        keys[k] = ((uint64_t) h.x << 33) | ((uint64_t) h.side << 32) | h.p;
    } else if (k < nl + ns) {
        lsg_hit_small const h = hs[k - nl];
        keys[k] = ((uint64_t) h.x << 33) | ((uint64_t) h.side << 32) | h.p;
    }
}

/* after the cofactor filter: the kept survivors, sorted by position, and
 * only the hits at them, so that the host gets no more than it uses.
 * Keys (x << 32) | (S0 << 24) | (S1 << 16), UINT64_MAX for a dropped one;
 * counts[0] = the number kept. */
__global__ void lsg_pack_kept(const lsg_survivor * surv, uint32_t n,
        uint64_t * keys, uint32_t * counts)
{
    uint32_t const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    lsg_survivor const v = surv[k];
    if (v.pad) {
        keys[k] = UINT64_MAX;
        return;
    }
    keys[k] = ((uint64_t) v.x << 32) | ((uint64_t) v.S[0] << 24) | ((uint64_t) v.S[1] << 16);
    atomicAdd(counts, 1u);
}

__global__ void lsg_unpack_kept(const uint64_t * keys, const uint32_t * counts,
        lsg_survivor * surv)
{
    uint32_t const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= counts[0]) return;
    uint64_t const e = keys[k];
    lsg_survivor v;
    v.x = (uint32_t) (e >> 32); v.S[0] = (uint8_t) (e >> 24); v.S[1] = (uint8_t) (e >> 16);
    v.pad = 0;
    surv[k] = v;
}

__device__ __forceinline__ bool lsg_is_kept(const uint64_t * keys, uint32_t n, uint32_t x)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t const mid = (lo + hi) / 2;
        if ((uint32_t) (keys[mid] >> 32) < x) lo = mid + 1; else hi = mid;
    }
    return lo < n && (uint32_t) (keys[lo] >> 32) == x;
}

template<typename H>
__global__ void lsg_keep_hits(const H * in, uint32_t n, const uint64_t * keys,
        const uint32_t * counts, H * out, uint32_t * nout)
{
    uint32_t const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    H const h = in[k];
    if (!lsg_is_kept(keys, counts[0], h.x)) return;
    out[atomicAdd(nout, 1u)] = h;
}

/* a warp per survivor, its lanes over the small sieve entries (one
 * thread per survivor left most of the GPU idle: 8% occupancy) */
/* the survivors are surv[0 .. min(*count, cap_surv) - first): their
 * number is read on the device, so that the host need not wait for it */
__global__ void lsg_resieve_small(const lsg_survivor * surv,
        const uint32_t * count, uint32_t first, uint32_t cap_surv,
        const lsg_small * E, uint32_t n, int logI, uint32_t side,
        lsg_hit_small * hits, uint32_t * nhits, uint32_t cap)
{
    /* a fixed grid whose warps go over the survivors (their number is
     * only on the device: a grid sized for the capacity launched mostly
     * warps with nothing to do) */
    uint32_t const lane = threadIdx.x & 31;
    uint32_t const nsurv = min(*count, cap_surv) - first;
    uint32_t const nwarps = (gridDim.x * blockDim.x) >> 5;
    for (uint32_t t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; t < nsurv; t += nwarps) {
    uint32_t const x = surv[t].x;
    uint32_t const ir = x & ((1u << logI) - 1);   /* i + I/2 */
    uint32_t const j = x >> logI;
    uint32_t const hI = 1u << (logI - 1);
    for (uint32_t e = lane; e < n; e += 32) {
        lsg_small const s = E[e];
        if (!s.resieve) continue;
        bool hit;
        uint32_t jj = j;
        if (s.kind != 0) {
            if (j % s.g) continue;
            jj = j / s.g;
        }
        uint64_t const v = (uint64_t) s.r * jj + hI;
        uint32_t const rv = v >> 32 ? (uint32_t) (v % s.q) : (uint32_t) v % s.q;
        hit = ir % s.q == rv;
        if (!hit) continue;
        uint32_t const k = atomicAdd(nhits, 1u);
        if (k < cap) {
            lsg_hit_small h;
            h.x = x; h.p = s.kind == 0 ? s.q : s.g; h.side = side;
            hits[k] = h;
        }
    }
    }
}
/* }}} */

/* {{{ host */
struct lsg_ctx {
    int device = 0;
    int nsm = 1;
    int slab_log = SLAB_LOG;    /* positions sieved at a time, log2 */
    bool slab_log_set = false;  /* by CADO_LAS_GPU_SLAB_LOG */
    cudaStream_t stream;
    /* waits on the stream sleep on this event instead of spinning in
     * cudaStreamSynchronize: with a dozen subjobs per GPU, the spinning
     * threads took 3/4 of las's CPU time, much of it from hyperthreads
     * whose siblings were cofactoring. CADO_LAS_GPU_SPIN=1: spin. */
    cudaEvent_t ev_wait;
    bool spin = false;
    uint8_t * d_norms[2] = { nullptr, nullptr }; size_t cap_norms[2] = { 0, 0 };
    uint32_t * d_SS[2] = { nullptr, nullptr }; size_t cap_SS[2] = { 0, 0 };
    double * d_Gs[2] = { nullptr, nullptr }; size_t cap_Gs[2] = { 0, 0 };
    uint8_t * d_S[2] = { nullptr, nullptr }; size_t cap_S[2] = { 0, 0 };
    lsg_lat * d_lats[2] = { nullptr, nullptr }; size_t cap_lats[2] = { 0, 0 };
    lsg_small * d_small[2] = { nullptr, nullptr }; size_t cap_small[2] = { 0, 0 };
    uint32_t * d_bitmap = nullptr; size_t cap_bitmap = 0;
    uint32_t * d_summary = nullptr;
    uint32_t * d_accept = nullptr;      /* 65536 bits, see lsg_sieve */
    /* the cofactor filter: hit keys (sorted into keys2), CUB's scratch,
     * the trial division primes */
    uint64_t * d_keys = nullptr, * d_keys2 = nullptr; size_t cap_keys = 0;
    /* and the compaction after it */
    uint64_t * d_skeys = nullptr, * d_skeys2 = nullptr; size_t cap_skeys = 0;
    lsg_hit_large * d_hl2 = nullptr; size_t cap_hl2 = 0;
    lsg_hit_small * d_hs2 = nullptr; size_t cap_hs2 = 0;
    uint32_t * d_kcounts = nullptr, * h_kcounts = nullptr;
    size_t last_rejected = 0;   /* by the cofactor filter, this call */
    void * d_cub = nullptr; size_t cap_cub = 0;
    uint32_t * d_td[2] = { nullptr, nullptr }; size_t cap_td[2] = { 0, 0 };
    uint64_t td_version = 0;
    
    uint64_t accept_version = 0;
    uint64_t * d_x[2][2] = { { nullptr, nullptr }, { nullptr, nullptr } };
    size_t cap_x[2][2] = { { 0, 0 }, { 0, 0 } };
    lsg_cst * d_cst[2] = { nullptr, nullptr }; size_t cap_cst[2] = { 0, 0 };
    uint32_t * d_counts = nullptr;      /* survivors, large hits, small hits */
    uint32_t * h_counts = nullptr;
    lsg_survivor * d_surv = nullptr; size_t cap_surv = 0;
    lsg_hit_large * d_hl = nullptr; size_t cap_hl = 0;
    lsg_hit_small * d_hs = nullptr; size_t cap_hs = 0;
    lsg_survivor * h_surv = nullptr; size_t cap_h_surv = 0;
    lsg_hit_large * h_hl = nullptr; size_t cap_h_hl = 0;
    lsg_hit_small * h_hs = nullptr; size_t cap_h_hs = 0;
    double seconds = 0;
    unsigned long calls = 0;
};

template<typename T> static void grow(T *& p, size_t & cap, size_t n)
{
    if (n <= cap) return;
    cudaFree(p);
    cap = n + n / 8 + 16;
    CK(cudaMalloc(&p, cap * sizeof(T)));
}
/* grows, keeping the first `keep` elements */
template<typename T> static void grow_keep(T *& p, size_t & cap, size_t n,
        size_t keep, cudaStream_t stream)
{
    if (n <= cap) return;
    T * q;
    size_t const ncap = n + n / 4 + 16;
    CK(cudaMalloc(&q, ncap * sizeof(T)));
    if (keep) CK(cudaMemcpyAsync(q, p, keep * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CK(cudaStreamSynchronize(stream));
    cudaFree(p);
    p = q;
    cap = ncap;
}
template<typename T> static void grow_host(T *& p, size_t & cap, size_t n)
{
    if (n <= cap) return;
    cudaFreeHost(p);
    cap = n + n / 8 + 16;
    CK(cudaMallocHost(&p, cap * sizeof(T)));
}

struct lsg_table {
    lsg_lat * d = nullptr;
    size_t n = 0;
};

struct lsg_table * lsg_table_create(const struct lsg_lat * lats, size_t n)
{
    auto * t = new lsg_table;
    t->n = n;
    if (n) {
        CK(cudaMalloc(&t->d, n * sizeof(lsg_lat)));
        CK(cudaMemcpy(t->d, lats, n * sizeof(lsg_lat), cudaMemcpyHostToDevice));
    }
    return t;
}

void lsg_table_free(struct lsg_table * t)
{
    if (!t) return;
    cudaFree(t->d);
    delete t;
}

/* the environment variable is set to 0 */
static bool getenv_flag_off(const char * name)
{
    const char * e = getenv(name);
    return e && strcmp(e, "0") == 0;
}

/* the environment variable is set to something other than 0 */
static bool getenv_flag(const char * name)
{
    const char * e = getenv(name);
    return e && strcmp(e, "0") != 0;
}

/* everything queued on c->stream so far is done */
static void lsg_wait(lsg_ctx * c)
{
    CK(cudaEventRecord(c->ev_wait, c->stream));
    CK(cudaEventSynchronize(c->ev_wait));
}

int lsg_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

struct lsg_ctx * lsg_ctx_create(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    auto * c = new lsg_ctx;
    const char * e = getenv("CADO_LAS_GPU_DEVICE");
    if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!e) e = getenv("SLURM_LOCALID");
    c->device = (e ? atoi(e) : cado_gpu_claim("las", ndev)) % ndev;
    CK(cudaSetDevice(c->device));
    if (!cado_cuda_can_run("las GPU siever")) { delete c; return nullptr; }
    CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    {
        char const * sp = getenv("CADO_LAS_GPU_SPIN");
        c->spin = sp && strcmp(sp, "0") != 0;
    }
    CK(cudaEventCreateWithFlags(&c->ev_wait, cudaEventDisableTiming
                | (c->spin ? 0 : cudaEventBlockingSync)));
    CK(cudaDeviceGetAttribute(&c->nsm, cudaDevAttrMultiProcessorCount, c->device));
    {
        /* SLAB_LOG was best with the 64 MB L2 of the RTX PRO 4500 (16
         * subjobs: 2^22 against 2^21 and 2^23); other GPUs get one more
         * for each doubling of the L2 (rounded: RTX PRO 6000 and B200, 
         * 128 and 126 MB, get 2^23). That is an estimate, not measured
         * on them; CADO_LAS_GPU_SLAB_LOG sets it. */
        int l2 = 0;
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, c->device);
        int sl = SLAB_LOG;
        if (l2 > 0)
            sl += (int) lround(log2((double) l2 / (64.0 * 1024 * 1024)));
        if (const char * e = getenv("CADO_LAS_GPU_SLAB_LOG"))
            if (atoi(e) > 0) { sl = atoi(e); c->slab_log_set = true; }
        c->slab_log = std::max(SLAB_LOG_MIN, std::min(SLAB_LOG_MAX, sl));
    }
    CK(cudaMalloc(&c->d_counts, 4 * sizeof(uint32_t)));
    CK(cudaMalloc(&c->d_summary, SUMMARY_WORDS * sizeof(uint32_t)));
    CK(cudaMallocHost(&c->h_counts, 4 * sizeof(uint32_t)));
    size_t const sh = (size_t(1) << TILE_LOG) * 2;
    CK(cudaFuncSetAttribute(lsg_small_tiles,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sh));
    return c;
}

void lsg_ctx_free(struct lsg_ctx * c)
{
    if (!c) return;
    cudaSetDevice(c->device);
    for (int k = 0; k < 2; k++) {
        cudaFree(c->d_norms[k]); cudaFree(c->d_SS[k]); cudaFree(c->d_S[k]);
        cudaFree(c->d_Gs[k]); cudaFree(c->d_x[k][0]); cudaFree(c->d_x[k][1]); cudaFree(c->d_cst[k]);
    }
    for (int k = 0; k < 2; k++) { cudaFree(c->d_lats[k]); cudaFree(c->d_small[k]); }
    cudaFree(c->d_bitmap); cudaFree(c->d_counts); cudaFree(c->d_summary);
    cudaFree(c->d_accept);
    cudaFree(c->d_keys); cudaFree(c->d_keys2); cudaFree(c->d_cub);
    cudaFree(c->d_skeys); cudaFree(c->d_skeys2); cudaFree(c->d_hl2); cudaFree(c->d_hs2);
    cudaFree(c->d_kcounts); cudaFreeHost(c->h_kcounts);
    cudaFree(c->d_td[0]); cudaFree(c->d_td[1]);
    cudaFree(c->d_surv); cudaFree(c->d_hl); cudaFree(c->d_hs);
    cudaFreeHost(c->h_counts); cudaFreeHost(c->h_surv); cudaFreeHost(c->h_hl);
    cudaFreeHost(c->h_hs);
    cudaEventDestroy(c->ev_wait);
    cudaStreamDestroy(c->stream);
    delete c;
}

void lsg_sieve(struct lsg_ctx * c, const struct lsg_qinfo * Qi,
        const struct lsg_side * sides, int nsides, uint8_t ** S,
        const uint8_t * bound, struct lsg_out * out,
        const uint32_t * accept, uint64_t accept_version,
        const struct lsg_cofilter * cof)
{
    CK(cudaSetDevice(c->device));
    c->last_rejected = 0;
    if (accept && accept_version != c->accept_version) {
        if (!c->d_accept)
            CK(cudaMalloc(&c->d_accept, 65536 / 8));
        CK(cudaMemcpyAsync(c->d_accept, accept, 65536 / 8, cudaMemcpyHostToDevice,
                    c->stream));
        c->accept_version = accept_version;
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, c->stream));

    int const logI = Qi->logI;
    size_t const area = (size_t) Qi->J << logI;
    /* slabs of whole rows (logI <= 16 < slab_log), whole tiles. Every
     * lattice's walk state is read and written for each slab, so a larger
     * factor base wants larger slabs, the L2 smaller ones: half a power
     * of 2 per doubling of the bucket-sieved entries, from the 426,000 of
     * the c120 where 2^22 was best (c140: 1.63 million, best 2^23, 11%
     * faster than 2^22; 2^21 and 2^24 slower) */
    int slab_log = c->slab_log;
    if (!c->slab_log_set) {
        size_t nt = 0;
        for (int side = 0; side < nsides; side++)
            nt += sides[side].table ? sides[side].table->n : 0;
        if (nt)
            slab_log += (int) lround(0.5 * log2((double) nt / 426000.0));
        slab_log = std::max(SLAB_LOG_MIN, std::min(SLAB_LOG_MAX, slab_log));
    }
    size_t const slab = std::min(area, (size_t) 1 << slab_log);

    lfg_qinfo Q {};
    Q.a0 = Qi->a0; Q.b0 = Qi->b0; Q.a1 = Qi->a1; Q.b1 = Qi->b1;
    for (int k = 0; k < Qi->nqfac; k++) Q.qfac[k] = Qi->qfac[k];
    Q.nqfac = Qi->nqfac;
    Q.logI = logI;
    Q.J = Qi->J;

    /* per side: inputs of this q, buffers, walk states (table, then the
     * extra lattices) */
    uint32_t nwe[2] = { 0, 0 };
    size_t ntab[2] = { 0, 0 }, nlat[2] = { 0, 0 };
    for (int side = 0; side < nsides; side++) {
        lsg_side const & sd = sides[side];
        grow(c->d_norms[side], c->cap_norms[side], slab);
        grow(c->d_SS[side], c->cap_SS[side], slab / 2);
        grow(c->d_S[side], c->cap_S[side], slab);
        size_t const ng = sd.norm.degree > 1 ? 3 * sd.norm.nseg + 1 : 1;
        grow(c->d_Gs[side], c->cap_Gs[side], ng);
        if (sd.norm.degree > 1) {
            CK(cudaMemcpyAsync(c->d_Gs[side], sd.G_endpoints, (sd.norm.nseg + 1) * sizeof(double),
                        cudaMemcpyHostToDevice, c->stream));
            CK(cudaMemcpyAsync(c->d_Gs[side] + sd.norm.nseg + 1, sd.G_equations,
                        2 * sd.norm.nseg * sizeof(double), cudaMemcpyHostToDevice, c->stream));
        }
        ntab[side] = sd.table ? sd.table->n : 0;
        nlat[side] = sd.nlats;
        grow(c->d_lats[side], c->cap_lats[side], sd.nlats + 1);
        if (sd.nlats)
            CK(cudaMemcpyAsync(c->d_lats[side], sd.lats, sd.nlats * sizeof(lsg_lat),
                        cudaMemcpyHostToDevice, c->stream));
        grow(c->d_small[side], c->cap_small[side], sd.nsmall + 1);
        if (sd.nsmall)
            CK(cudaMemcpyAsync(c->d_small[side], sd.small, sd.nsmall * sizeof(lsg_small),
                        cudaMemcpyHostToDevice, c->stream));
        uint32_t const rowlen = logI >= TILE_LOG ? (1u << TILE_LOG) : (1u << logI);
        while (nwe[side] < sd.nsmall && sd.small[nwe[side]].q * 64 <= rowlen) nwe[side]++;
        for (int k = 0; k < 2; k++)
            grow(c->d_x[side][k], c->cap_x[side][k], ntab[side] + nlat[side] + 1);
        grow(c->d_cst[side], c->cap_cst[side], ntab[side] + nlat[side] + 1);
    }

    bool const want_out = out && nsides == 2;
    size_t const nwords = (slab + 31) / 32;
    int glog = 6;
    while ((slab >> glog) > (size_t) SUMMARY_WORDS * 32) glog++;
    uint32_t const nsummary = (uint32_t) (((slab >> glog) + 31) / 32);
    if (want_out) {
        grow(c->d_bitmap, c->cap_bitmap, nwords);
        CK(cudaMemsetAsync(c->d_counts, 0, 4 * sizeof(uint32_t), c->stream));
        grow(c->d_surv, c->cap_surv, 65536);
        grow(c->d_hl, c->cap_hl, 1 << 20);
        grow(c->d_hs, c->cap_hs, 1 << 20);
    }
    uint32_t done[3] = { 0, 0, 0 };   /* survivors, large hits, small hits so far */

    /* the resieving of a slab and the sieving of the next in one pass
     * (lsg_resieve_sieve); CADO_LAS_GPU_FUSE=0: separately */
    bool const fuse = want_out && !getenv_flag_off("CADO_LAS_GPU_FUSE");
    for (size_t lo = 0; lo < area; lo += slab) {
        size_t const hi = std::min(area, lo + slab);
        size_t const hi2 = std::min(area, hi + slab);   /* the next slab's end */
        size_t const npos = hi - lo;
        uint32_t const j_lo = (uint32_t) (lo >> logI);
        uint32_t const nrows = (uint32_t) (npos >> logI);
        bool const first = lo == 0;
        for (int side = 0; side < nsides; side++) {
            lsg_side const & sd = sides[side];
            uint8_t * const norms = c->d_norms[side];
            uint32_t * const SS = c->d_SS[side];
            double * const G = c->d_Gs[side];
            if (sd.norm.degree > 1 && sd.norm.nseg > 1 && !getenv_flag("CADO_LAS_GPU_NORMS_ROW")) {
                size_t const nt = (size_t) sd.norm.nseg * nrows;
                lsg_norms_alg<<<(unsigned) ((nt + 63) / 64), 64, 0, c->stream>>>(norms,
                        sd.norm, G, G + sd.norm.nseg + 1, logI, j_lo, nrows);
            } else {
                lsg_norms<<<(nrows + 63) / 64, 64, 0, c->stream>>>(norms, sd.norm,
                        G, G + sd.norm.nseg + 1, logI, j_lo, nrows);
            }
            /* with fusing, the previous slab's resieving pass did this
             * slab's sums already */
            if (!(fuse && !first)) {
            CK(cudaMemsetAsync(SS, 0, npos / 2 * sizeof(uint32_t), c->stream));
            /* the table's lattices, then this q's: x and cst in that order;
             * without a resieving pass, lsg_large writes x_next itself */
            uint64_t * const xc = c->d_x[side][0];
            uint64_t * const xn = want_out ? nullptr : c->d_x[side][1];
            lsg_cst * const cs = c->d_cst[side];
            if (ntab[side])
                lsg_large<<<(unsigned) ((ntab[side] + 127) / 128), 128, 0, c->stream>>>(
                        sd.table->d, (uint32_t) ntab[side], Q, first, xc, cs, xn,
                        lo, hi, SS);
            if (nlat[side])
                lsg_large<<<(unsigned) ((nlat[side] + 127) / 128), 128, 0, c->stream>>>(
                        c->d_lats[side], (uint32_t) nlat[side], Q, first,
                        xc + ntab[side], cs + ntab[side], xn ? xn + ntab[side] : nullptr,
                        lo, hi, SS);
            }
            lsg_small_tiles<<<(unsigned) (npos >> TILE_LOG), 512, (size_t(1) << TILE_LOG) * 2, c->stream>>>(
                    c->d_small[side], (uint32_t) sd.nsmall, nwe[side], logI, lo, SS, norms,
                    c->d_S[side]);
            CK(cudaGetLastError());
            if (S && S[side])
                CK(cudaMemcpyAsync(S[side] + lo, getenv("CADO_LAS_GPU_SIEVE_NORMS") ?
                            norms : c->d_S[side], npos, cudaMemcpyDeviceToHost,
                            c->stream));
        }

        if (want_out) {
            /* survivors of the slab, and their resieving; redone with
             * more room if a list overflows */
            for (;;) {
                CK(cudaMemsetAsync(c->d_bitmap, 0, nwords * sizeof(uint32_t), c->stream));
                CK(cudaMemsetAsync(c->d_summary, 0, SUMMARY_WORDS * sizeof(uint32_t), c->stream));
                CK(cudaMemcpyAsync(c->d_counts, done, 3 * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, c->stream));
                lsg_survivors<<<(unsigned) ((npos / 16 + 255) / 256), 256, 0, c->stream>>>(
                        c->d_S[0], c->d_S[1], lo, npos, logI, bound[0], bound[1],
                        c->d_surv, c->d_counts, (uint32_t) c->cap_surv, c->d_bitmap,
                        c->d_summary, glog, accept ? c->d_accept : nullptr);
                /* no wait for the survivor count here (it left the
                 * stream idle): the resieving kernels read it on the
                 * device, and every count is checked once, below */
                for (int side = 0; side < 2; side++) {
                    lsg_side const & sd = sides[side];
                    uint64_t * const xc = c->d_x[side][0];
                    uint64_t * const xn = c->d_x[side][1];
                    lsg_cst * const cs = c->d_cst[side];
                    auto const rl_blocks = [&](size_t nn) {
                        return (unsigned) std::min<size_t>((nn + 255) / 256, 8 * (size_t) c->nsm);
                    };
                    if (fuse && hi < area) {
                        /* and the next slab's sieving, into its cleared sums */
                        uint32_t * const SS2 = c->d_SS[side];
                        CK(cudaMemsetAsync(SS2, 0, (hi2 - hi) / 2 * sizeof(uint32_t), c->stream));
                        if (ntab[side])
                            lsg_resieve_sieve<<<rl_blocks(ntab[side]), 256, 0, c->stream>>>(
                                    sd.table->d, (uint32_t) ntab[side], Q, xc, cs, xn,
                                    lo, hi, hi2, c->d_bitmap, c->d_summary, glog, nsummary,
                                    side, c->d_hl, c->d_counts + 1, (uint32_t) c->cap_hl, SS2);
                        if (nlat[side])
                            lsg_resieve_sieve<<<rl_blocks(nlat[side]), 256, 0, c->stream>>>(
                                    c->d_lats[side], (uint32_t) nlat[side], Q,
                                    xc + ntab[side], cs + ntab[side], xn + ntab[side],
                                    lo, hi, hi2, c->d_bitmap, c->d_summary, glog, nsummary,
                                    side, c->d_hl, c->d_counts + 1, (uint32_t) c->cap_hl, SS2);
                    } else {
                    if (ntab[side])
                        lsg_resieve_large<<<rl_blocks(ntab[side]), 256, 0, c->stream>>>(
                                sd.table->d, (uint32_t) ntab[side], Q, xc, cs, xn, lo, hi,
                                c->d_bitmap, c->d_summary, glog, nsummary, side,
                                c->d_hl, c->d_counts + 1, (uint32_t) c->cap_hl);
                    if (nlat[side])
                        lsg_resieve_large<<<rl_blocks(nlat[side]), 256, 0, c->stream>>>(
                                c->d_lats[side], (uint32_t) nlat[side], Q,
                                xc + ntab[side], cs + ntab[side], xn + ntab[side], lo, hi,
                                c->d_bitmap, c->d_summary, glog, nsummary, side,
                                c->d_hl, c->d_counts + 1, (uint32_t) c->cap_hl);
                    }
                    if (sd.nsmall && c->cap_surv > done[0])
                        lsg_resieve_small<<<(unsigned) std::min<size_t>((c->cap_surv - done[0] + 3) / 4,
                                    8 * (size_t) c->nsm), 128, 0, c->stream>>>(
                                c->d_surv + done[0], c->d_counts, done[0], (uint32_t) c->cap_surv,
                                c->d_small[side], (uint32_t) sd.nsmall, logI, side, c->d_hs,
                                c->d_counts + 2, (uint32_t) c->cap_hs);
                }
                CK(cudaGetLastError());
                CK(cudaMemcpyAsync(c->h_counts, c->d_counts, 3 * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, c->stream));
                lsg_wait(c);
                bool redo = false;
                if (c->h_counts[0] > c->cap_surv) {
                    size_t cap = c->h_counts[0];
                    grow_keep(c->d_surv, c->cap_surv, cap, done[0], c->stream);
                    redo = true;
                }
                if (c->h_counts[1] > c->cap_hl) {
                    size_t cap = c->h_counts[1];
                    grow_keep(c->d_hl, c->cap_hl, cap, done[1], c->stream);
                    redo = true;
                }
                if (c->h_counts[2] > c->cap_hs) {
                    size_t cap = c->h_counts[2];
                    grow_keep(c->d_hs, c->cap_hs, cap, done[2], c->stream);
                    redo = true;
                }
                if (redo) continue;
                for (int k = 0; k < 3; k++) done[k] = c->h_counts[k];
                break;
            }
        }
        /* the end states of this slab start the next one */
        for (int side = 0; side < nsides; side++)
            std::swap(c->d_x[side][0], c->d_x[side][1]);
    }
    /* the cofactor filter: marks (pad = 1) the survivors that
     * check_leftover_norm() rejects, so that the CPU skips them */
    if (want_out && cof && cof->bits && done[0]) {
        uint32_t const nkeys = done[1] + done[2];
        if (nkeys + 1 > c->cap_keys) {
            cudaFree(c->d_keys); cudaFree(c->d_keys2);
            c->cap_keys = nkeys + nkeys / 4 + 1024;
            CK(cudaMalloc(&c->d_keys, c->cap_keys * sizeof(uint64_t)));
            CK(cudaMalloc(&c->d_keys2, c->cap_keys * sizeof(uint64_t)));
        }
        if (nkeys)
            lsg_hit_keys<<<(nkeys + 255) / 256, 256, 0, c->stream>>>(
                    c->d_hl, done[1], c->d_hs, done[2], c->d_keys);
        size_t need = 0;
        CK(cub::DeviceRadixSort::SortKeys(nullptr, need, c->d_keys, c->d_keys2,
                    (int) nkeys, 0, 64, c->stream));
        if (need > c->cap_cub) {
            cudaFree(c->d_cub);
            c->cap_cub = need + need / 4;
            CK(cudaMalloc(&c->d_cub, c->cap_cub));
        }
        size_t have = c->cap_cub;
        if (nkeys)
            CK(cub::DeviceRadixSort::SortKeys(c->d_cub, have, c->d_keys, c->d_keys2,
                        (int) nkeys, 0, 64, c->stream));
        lsg_cofilter P = *cof;
        for (int side = 0; side < 2; side++) {
            if (cof->td_version != c->td_version) {
                grow(c->d_td[side], c->cap_td[side], cof->s[side].ntd + 1);
                if (cof->s[side].ntd)
                    CK(cudaMemcpyAsync(c->d_td[side], cof->s[side].td,
                                cof->s[side].ntd * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, c->stream));
            }
            P.s[side].td = c->d_td[side];
        }
        c->td_version = cof->td_version;
        lsg_cofilter_launch(c->stream, P, c->d_surv, done[0], c->d_keys2,
                nkeys, logI);
        CK(cudaGetLastError());

        /* keep only the survivors not rejected, and their hits */
        uint32_t const ns = done[0];
        if (ns + 1 > c->cap_skeys) {
            cudaFree(c->d_skeys); cudaFree(c->d_skeys2);
            c->cap_skeys = ns + ns / 4 + 1024;
            CK(cudaMalloc(&c->d_skeys, c->cap_skeys * sizeof(uint64_t)));
            CK(cudaMalloc(&c->d_skeys2, c->cap_skeys * sizeof(uint64_t)));
        }
        grow(c->d_hl2, c->cap_hl2, done[1] + 1);
        grow(c->d_hs2, c->cap_hs2, done[2] + 1);
        if (!c->d_kcounts) {
            CK(cudaMalloc(&c->d_kcounts, 4 * sizeof(uint32_t)));
            CK(cudaMallocHost(&c->h_kcounts, 4 * sizeof(uint32_t)));
        }
        CK(cudaMemsetAsync(c->d_kcounts, 0, 4 * sizeof(uint32_t), c->stream));
        lsg_pack_kept<<<(ns + 255) / 256, 256, 0, c->stream>>>(c->d_surv, ns,
                c->d_skeys, c->d_kcounts);
        need = 0;
        CK(cub::DeviceRadixSort::SortKeys(nullptr, need, c->d_skeys, c->d_skeys2,
                    (int) ns, 0, 64, c->stream));
        if (need > c->cap_cub) {
            cudaFree(c->d_cub);
            c->cap_cub = need + need / 4;
            CK(cudaMalloc(&c->d_cub, c->cap_cub));
        }
        have = c->cap_cub;
        CK(cub::DeviceRadixSort::SortKeys(c->d_cub, have, c->d_skeys, c->d_skeys2,
                    (int) ns, 0, 64, c->stream));
        lsg_unpack_kept<<<(ns + 255) / 256, 256, 0, c->stream>>>(c->d_skeys2,
                c->d_kcounts, c->d_surv);
        if (done[1])
            lsg_keep_hits<<<(done[1] + 255) / 256, 256, 0, c->stream>>>(c->d_hl,
                    done[1], c->d_skeys2, c->d_kcounts, c->d_hl2, c->d_kcounts + 1);
        if (done[2])
            lsg_keep_hits<<<(done[2] + 255) / 256, 256, 0, c->stream>>>(c->d_hs,
                    done[2], c->d_skeys2, c->d_kcounts, c->d_hs2, c->d_kcounts + 2);
        CK(cudaGetLastError());
        CK(cudaMemcpyAsync(c->h_kcounts, c->d_kcounts, 3 * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, c->stream));
        lsg_wait(c);
        c->last_rejected = done[0] - c->h_kcounts[0];
        done[0] = c->h_kcounts[0];
        done[1] = c->h_kcounts[1];
        done[2] = c->h_kcounts[2];
        std::swap(c->d_hl, c->d_hl2); std::swap(c->cap_hl, c->cap_hl2);
        std::swap(c->d_hs, c->d_hs2); std::swap(c->cap_hs, c->cap_hs2);
    }
    /* the copies below are ordered after the kernels on the stream */
    CK(cudaEventRecord(e1, c->stream));

    if (want_out) {
        uint32_t const nsurv = done[0], nl = done[1], ns = done[2];
        grow_host(c->h_surv, c->cap_h_surv, nsurv + 1);
        grow_host(c->h_hl, c->cap_h_hl, nl + 1);
        grow_host(c->h_hs, c->cap_h_hs, ns + 1);
        if (nsurv) CK(cudaMemcpyAsync(c->h_surv, c->d_surv, nsurv * sizeof(lsg_survivor),
                    cudaMemcpyDeviceToHost, c->stream));
        if (nl) CK(cudaMemcpyAsync(c->h_hl, c->d_hl, nl * sizeof(lsg_hit_large),
                    cudaMemcpyDeviceToHost, c->stream));
        if (ns) CK(cudaMemcpyAsync(c->h_hs, c->d_hs, ns * sizeof(lsg_hit_small),
                    cudaMemcpyDeviceToHost, c->stream));
        out->surv = c->h_surv; out->nsurv = nsurv;
        out->nrejected = c->last_rejected;
        out->large = c->h_hl; out->nlarge = nl;
        out->small = c->h_hs; out->nsmall = ns;
    }

    lsg_wait(c);
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    c->seconds += ms * 1e-3;
    c->calls++;
}

void lsg_ctx_stats(const struct lsg_ctx * c, double * seconds, unsigned long * calls)
{
    *seconds = c->seconds;
    *calls = c->calls;
}
/* }}} */

/* vim: set sw=4 fdm=marker: */
