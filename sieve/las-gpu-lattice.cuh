#ifndef CADO_LAS_GPU_LATTICE_CUH
#define CADO_LAS_GPU_LATTICE_CUH

/* Device code shared by the GPU bucket fill (las-fill-gpu-kernels.cu)
 * and the GPU siever (las-sieve-gpu-kernels.cu): the root transform of
 * las-fbroot-qlattice.hpp and the p-lattices of las-plattice.hpp /
 * las-reduce-plattice-simplistic.hpp. */

#include <cstdint>

#include "las-fill-gpu-kernels.h"

/* a p-lattice walk in progress (plattice_enumerator) */
struct lfg_state {
    uint64_t x;             /* UINT64_MAX: no walk */
    uint64_t inc_warp, inc_step;
    uint32_t bound_step, bound_warp;
};

/* {{{ device: arithmetic of las-fbroot-qlattice.hpp / las-plattice.hpp */
static __device__ __forceinline__ uint32_t mod_signed(int64_t a, uint32_t p)
{
    int64_t const r = a % (int64_t) p;
    return (uint32_t) (r < 0 ? r + p : r);
}

/* 1/a mod p, 0 if not invertible */
static __device__ uint32_t inv_mod(uint32_t a, uint32_t p)
{
    if (a == 0) return 0;
    int64_t r0 = p, r1 = a, u0 = 0, u1 = 1;
    while (r1) {
        int64_t const q = r0 / r1;
        int64_t t = r0 - q * r1; r0 = r1; r1 = t;
        t = u0 - q * u1; u0 = u1; u1 = t;
    }
    if (r0 != 1) return 0;
    return (uint32_t) (u0 < 0 ? u0 + p : u0);
}

/* xgcd_ul of utils/gcd.c */
static __device__ unsigned long xgcd_ul(unsigned long * xa, unsigned long a, unsigned long b)
{
    if (b == 0) { *xa = 1; return a; }
    if (b == a) { *xa = 0; return a; }
    unsigned long r0 = b, r1 = a, u0 = 0, u1 = 1;
    unsigned int i = 0;
    for (; r1 > 0; i++) {
        unsigned long const q = r0 / r1;
        unsigned long const r2 = r0 - q * r1;
        unsigned long const u2 = u0 + q * u1;
        r0 = r1; r1 = r2;
        u0 = u1; u1 = u2;
    }
    if ((i & 1U) == 0 && u0) u0 = b / r0 - u0;
    *xa = u0;
    return r0;
}

struct pli_t { uint32_t mi0, j0, i1, j1; };

static __device__ void reduce_with_vertical_vector(pli_t & L, uint32_t I)
{
    uint32_t const a = (I - 1) % L.mi0;
    uint32_t mi2 = (I - 1) - a - L.i1;
    uint32_t const j2 = L.j1;
    if (L.mi0 + mi2 < I) mi2 += L.mi0;
    L.i1 = L.mi0;
    L.j1 = L.j0;
    L.mi0 = mi2;
    L.j0 = j2;
}

/* reduce_plattice_simplistic */
static __device__ void reduce(pli_t & L, uint32_t I)
{
    if (L.i1 == 0 || (L.j1 > 1 && L.mi0 < I)) {
        reduce_with_vertical_vector(L, I);
        return;
    }
    for (;;) {
        if (L.i1 < I) {
            if (L.i1 == 0) {
                L.j0 = L.j1 - L.j0;
                reduce_with_vertical_vector(L, I);
                return;
            }
            int const a = (L.mi0 + L.i1 - I) / L.i1;
            L.mi0 -= a * L.i1;
            L.j0 += a * L.j1;
            return;
        }
        {
            int const k = L.mi0 / L.i1;
            L.mi0 -= k * L.i1;
            L.j0 += k * L.j1;
        }
        if (L.mi0 < I) {
            if (L.mi0 == 0) {
                L.mi0 = L.i1;
                L.i1 = L.j0; L.j0 = L.j1; L.j1 = L.i1;
                L.i1 = 0;
                reduce_with_vertical_vector(L, I);
                return;
            }
            int const a = (L.mi0 + L.i1 - I) / L.mi0;
            L.i1 -= a * L.mi0;
            L.j1 += a * L.j0;
            return;
        }
        {
            int const k = L.i1 / L.mi0;
            L.i1 -= k * L.mi0;
            L.j1 += k * L.j0;
        }
    }
}

/* Everything up to the start of the walk. Returns 0 (nothing to do),
 * 1 (walk, state set) or 2 (special lattice, sp set). */
static __device__ int setup_lattice(lfg_lat const & lat, lfg_qinfo const & Q,
        lfg_state & st, lfg_special & sp)
{
    for (int k = 0; k < Q.nqfac; k++)
        if (Q.qfac[k] == lat.p) return 0;
    uint32_t const p = lat.q;   /* the modulus: p^k */

    /* fb_root_in_qlattice_31bits */
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
    uint32_t r;
    bool proj;
    uint32_t iv = inv_mod(v, p);
    if (iv) {
        r = (uint32_t) (((uint64_t) u * iv) % p);
        proj = false;
    } else {
        uint32_t const iu = inv_mod(u, p);
        if (!iu) return 0;      /* the CPU code aborts there */
        r = (uint32_t) (((uint64_t) v * iu) % p);
        proj = true;
    }

    /* plattice_info(q, r, proj, logI) */
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

    /* plattice_enumerator */
    int const logI = Q.logI;
    st.inc_warp = ((uint64_t) L.j0 << logI) + (int64_t) (-(int32_t) L.mi0);
    st.inc_step = (L.i1 >> logI) ? UINT64_MAX
                : ((uint64_t) L.j1 << logI) + (int64_t) L.i1;
    st.bound_step = L.mi0;
    st.bound_warp = (L.i1 >> logI) ? 0 : I - L.i1;
    st.x = (uint64_t) 1 << (logI - 1);
    /* skip (0,0) */
    {
        uint32_t const i = st.x & (I - 1);
        if (i >= st.bound_warp) st.x += st.inc_warp;
        if (i < st.bound_step) st.x += st.inc_step;
    }
    if (L.mi0 == 0 && L.j0 == 0) return 0;      /* discarded */
    bool const projlike = !(st.inc_step >> logI);
    bool const vertical = L.i1 >> logI;
    if (projlike || vertical) {
        sp.q = p; sp.r = r; sp.hint = lat.hint; sp.slice = lat.slice;
        sp.proj = proj;
        return 2;
    }
    return 1;
}

static __device__ __forceinline__ void step(lfg_state & st, uint32_t maskI)
{
    uint32_t const i = st.x & maskI;
    if (i >= st.bound_warp) st.x += st.inc_warp;
    if (i < st.bound_step) st.x += st.inc_step;
}
/* }}} */


#endif /* CADO_LAS_GPU_LATTICE_CUH */
