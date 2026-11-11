/* GPU cofactorization for las: the facul chain of methods (P-1, P+1,
 * ECM) on one side of a candidate, one thread per (candidate, side).
 *
 * This follows the CPU code closely so that it finds the same factors:
 *
 *  - stage 1 of P+1 and ECM interprets CADO's own PRAC bytecode (the same
 *    for every thread of a launch, so there is no divergence), P-1 uses
 *    the plan's exponent;
 *  - stage 2 runs the plan of stage2.cpp: the same w, U list and pairs,
 *    with backtracking after each giant step as in pp1_stage2.hpp and
 *    ecm.cpp;
 *  - a found factor is handled as in facul_doit_onefm (B^2, B^3 and
 *    lpb tests, BPSW as arithxx is_prime), and the chain as facul_all /
 *    facul_aux do for one side.
 *
 * Arithmetic: Montgomery representation with W 64-bit words, W = 1 for
 * cofactors of at most 64 bits, W = 2 below 2^126. The ECM curve and the
 * P+1 start value come as rationals (facul-gpu.cpp) reduced mod n here.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"

#include "facul-gpu-data.h"
#include "facul-gpu-kernels.h"

typedef unsigned __int128 u128;

/* PRAC bytecode (bytecode.h) */
#define PRAC_SUBBLOCK_INIT 0x69
#define PRAC_SUBBLOCK_FINAL 0x66
#define PRAC_BLOCK_FINAL 0x46
#define PRAC_SWAP 0x73

/* {{{ u128 helpers */
__device__ __forceinline__ int u128_bits(u128 x)
{
    uint64_t const hi = (uint64_t)(x >> 64);
    if (hi) return 128 - __clzll((long long) hi);
    uint64_t const lo = (uint64_t) x;
    return lo ? 64 - __clzll((long long) lo) : 0;
}

__device__ __forceinline__ int u128_ctz(u128 x)
{
    uint64_t const lo = (uint64_t) x;
    if (lo) return __ffsll((long long) lo) - 1;
    return 64 + __ffsll((long long) (uint64_t) (x >> 64)) - 1;
}

/* mpz_get_d truncates; so does this (both parts are exact or rounded
 * towards zero, and the sum of two such parts can differ from the
 * truncation only beyond 2^53 relative precision). */
__device__ __forceinline__ double u128_to_double(u128 x)
{
    return __ull2double_rz((uint64_t)(x >> 64)) * 18446744073709551616.0
        + __ull2double_rz((uint64_t) x);
}

__device__ u128 u128_gcd(u128 a, u128 b)
{
    if (a == 0) return b;
    if (b == 0) return a;
    int const k = min(u128_ctz(a), u128_ctz(b));
    a >>= u128_ctz(a);
    do {
        b >>= u128_ctz(b);
        if (a > b) { u128 const t = a; a = b; b = t; }
        b -= a;
    } while (b);
    return a << k;
}

/* inverse of a mod n (n odd), binary extended Euclid. Returns the gcd;
 * *inv is valid if it is 1. */
__device__ u128 u128_modinv(u128 * inv, u128 a, u128 n)
{
    a %= n;
    u128 u = a, v = n, x1 = 1, x2 = 0;
    if (u == 0) { *inv = 0; return n; }
    while (u != 0) {
        while (!(u & 1)) {
            u >>= 1;
            /* x1 / 2 mod n; x1 + n < 2^128 as n < 2^127 */
            x1 = (x1 & 1) ? (x1 >> 1) + (n >> 1) + 1 : x1 >> 1;
        }
        while (!(v & 1)) {
            v >>= 1;
            x2 = (x2 & 1) ? (x2 >> 1) + (n >> 1) + 1 : x2 >> 1;
        }
        if (u >= v) {
            u -= v;
            x1 = x1 >= x2 ? x1 - x2 : x1 + (n - x2);
        } else {
            v -= u;
            x2 = x2 >= x1 ? x2 - x1 : x2 + (n - x1);
        }
    }
    *inv = x2;
    return v;
}

/* Jacobi symbol (a/n), n odd positive */
__device__ int u128_jacobi(u128 a, u128 n)
{
    a %= n;
    int t = 1;
    while (a != 0) {
        int const z = u128_ctz(a);
        a >>= z;
        unsigned const r = (unsigned) (n & 7);
        if ((z & 1) && (r == 3 || r == 5)) t = -t;
        if ((a & 3) == 3 && (n & 3) == 3) t = -t;
        u128 const s = a; a = n; n = s;
        a %= n;
    }
    return n == 1 ? t : 0;
}

__device__ bool u128_is_square(u128 x)
{
    double const d = sqrt(u128_to_double(x));
    u128 r = (u128) d;
    /* fix the rounding */
    while (r * r > x) r--;
    while ((r + 1) * (r + 1) <= x) r++;
    return r * r == x;
}
/* }}} */

/* {{{ Montgomery arithmetic with W words */
template<int W> struct Res { uint64_t v[W]; };

template<int W> struct Mod {
    uint64_t n[W];
    uint64_t ninv;      /* -1/n mod 2^64 */
    Res<W> one;         /* R mod n, R = 2^(64W) */
    Res<W> r2;          /* R^2 mod n */
};

template<int W> __device__ __forceinline__ u128 res_int(Res<W> const & a)
{
    if (W == 1) return a.v[0];
    return ((u128) a.v[W - 1] << 64) | a.v[0];
}

template<int W> __device__ __forceinline__ Res<W> int_res(u128 x)
{
    Res<W> r;
    r.v[0] = (uint64_t) x;
    if (W == 2) r.v[W - 1] = (uint64_t)(x >> 64);
    return r;
}

template<int W> __device__ __forceinline__ u128 mod_int(Mod<W> const & M)
{
    if (W == 1) return M.n[0];
    return ((u128) M.n[W - 1] << 64) | M.n[0];
}

template<int W> __device__ __forceinline__ bool is0(Res<W> const & a)
{
    uint64_t t = 0;
#pragma unroll
    for (int i = 0; i < W; i++) t |= a.v[i];
    return t == 0;
}

template<int W> __device__ __forceinline__ bool eq(Res<W> const & a, Res<W> const & b)
{
    uint64_t t = 0;
#pragma unroll
    for (int i = 0; i < W; i++) t |= a.v[i] ^ b.v[i];
    return t == 0;
}

template<int W>
__device__ __forceinline__ Res<W> add(Res<W> const & a, Res<W> const & b, Mod<W> const & M)
{
    Res<W> s, d;
    uint64_t c = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const t = (u128) a.v[i] + b.v[i] + c;
        s.v[i] = (uint64_t) t;
        c = (uint64_t)(t >> 64);
    }
    uint64_t br = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const t = (u128) s.v[i] - M.n[i] - br;
        d.v[i] = (uint64_t) t;
        br = (uint64_t)(t >> 64) & 1;
    }
    bool const use = c || !br;
#pragma unroll
    for (int i = 0; i < W; i++) s.v[i] = use ? d.v[i] : s.v[i];
    return s;
}

template<int W>
__device__ __forceinline__ Res<W> sub(Res<W> const & a, Res<W> const & b, Mod<W> const & M)
{
    Res<W> d, s;
    uint64_t br = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const t = (u128) a.v[i] - b.v[i] - br;
        d.v[i] = (uint64_t) t;
        br = (uint64_t)(t >> 64) & 1;
    }
    uint64_t c = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const t = (u128) d.v[i] + M.n[i] + c;
        s.v[i] = (uint64_t) t;
        c = (uint64_t)(t >> 64);
    }
#pragma unroll
    for (int i = 0; i < W; i++) d.v[i] = br ? s.v[i] : d.v[i];
    return d;
}

/* a*b/R mod n, CIOS. Needs a < R, b < n (or the reverse). */
template<int W>
__device__ __forceinline__ Res<W> mul(Res<W> const & a, Res<W> const & b, Mod<W> const & M)
{
    uint64_t t[W + 2];
#pragma unroll
    for (int i = 0; i < W + 2; i++) t[i] = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        uint64_t c = 0;
#pragma unroll
        for (int j = 0; j < W; j++) {
            u128 const p = (u128) a.v[i] * b.v[j] + t[j] + c;
            t[j] = (uint64_t) p;
            c = (uint64_t)(p >> 64);
        }
        u128 s = (u128) t[W] + c;
        t[W] = (uint64_t) s;
        t[W + 1] = (uint64_t)(s >> 64);
        uint64_t const m = t[0] * M.ninv;
        u128 p = (u128) m * M.n[0] + t[0];
        c = (uint64_t)(p >> 64);
#pragma unroll
        for (int j = 1; j < W; j++) {
            p = (u128) m * M.n[j] + t[j] + c;
            t[j - 1] = (uint64_t) p;
            c = (uint64_t)(p >> 64);
        }
        s = (u128) t[W] + c;
        t[W - 1] = (uint64_t) s;
        t[W] = t[W + 1] + (uint64_t)(s >> 64);
    }
    Res<W> r, d;
    uint64_t br = 0;
#pragma unroll
    for (int j = 0; j < W; j++) {
        u128 const x = (u128) t[j] - M.n[j] - br;
        d.v[j] = (uint64_t) x;
        br = (uint64_t)(x >> 64) & 1;
    }
    bool const use = t[W] || !br;
#pragma unroll
    for (int j = 0; j < W; j++) r.v[j] = use ? d.v[j] : t[j];
    return r;
}

template<int W>
__device__ __forceinline__ Res<W> sqr(Res<W> const & a, Mod<W> const & M)
{
    return mul(a, a, M);
}

template<int W> __device__ void mod_init(Mod<W> & M, u128 n)
{
    M.n[0] = (uint64_t) n;
    if (W == 2) M.n[W - 1] = (uint64_t)(n >> 64);
    uint64_t const n0 = M.n[0];
    uint64_t x = n0;    /* 1/n0 mod 2^3 */
#pragma unroll
    for (int i = 0; i < 5; i++) x *= 2 - n0 * x;
    M.ninv = -x;
    /* R mod n */
    u128 r;
    if (W == 1) {
        r = (uint64_t)(-n0) % n0;
    } else {
        r = ((u128) 0 - n) % n;
    }
    M.one = int_res<W>(r);
    Res<W> t = M.one;
    for (int i = 0; i < 64 * W; i++)
        t = add(t, t, M);
    M.r2 = t;
}

/* integer (< R) to Montgomery form */
template<int W>
__device__ __forceinline__ Res<W> to_mont(u128 x, Mod<W> const & M)
{
    return mul(int_res<W>(x), M.r2, M);
}

template<int W>
__device__ __forceinline__ u128 from_mont(Res<W> const & a, Mod<W> const & M)
{
    Res<W> one;
#pragma unroll
    for (int i = 0; i < W; i++) one.v[i] = 0;
    one.v[0] = 1;
    return res_int(mul(a, one, M));
}

template<int W>
__device__ __forceinline__ Res<W> small(uint64_t s, Mod<W> const & M)
{
    return to_mont<W>((u128) s, M);
}

/* gcd(a, n); the Montgomery factor R is coprime to n */
template<int W>
__device__ __forceinline__ u128 res_gcd(Res<W> const & a, Mod<W> const & M)
{
    return u128_gcd(res_int(a), mod_int(M));
}

/* r = 1/a. Returns false (and r undefined) if a is not invertible. */
template<int W>
__device__ bool inv(Res<W> & r, Res<W> const & a, Mod<W> const & M)
{
    u128 const n = mod_int(M);
    u128 x = from_mont(a, M), y;
    if (u128_modinv(&y, x, n) != 1) return false;
    r = to_mont<W>(y, M);
    return true;
}

/* a multi-word integer from the pool, reduced mod n */
template<int W>
__device__ Res<W> bigint_mod(fgpu_bigint const & z, const uint64_t * pool64, Mod<W> const & M)
{
    Res<W> acc;
#pragma unroll
    for (int i = 0; i < W; i++) acc.v[i] = 0;
    if (z.len == 0) return acc;
    Res<W> base;    /* 2^64 in Montgomery form */
    if (W == 1) {
        base = M.r2;
    } else {
        Res<W> t;
#pragma unroll
        for (int i = 0; i < W; i++) t.v[i] = 0;
        t.v[1] = 1;
        base = mul(t, M.r2, M);
    }
    for (int i = (int) z.len - 1; i >= 0; i--) {
        uint64_t const limb = __ldg(pool64 + z.off + i);
        Res<W> l;
#pragma unroll
        for (int j = 0; j < W; j++) l.v[j] = 0;
        l.v[0] = limb;
        acc = add(mul(acc, base, M), mul(l, M.r2, M), M);
    }
    if (z.sign < 0) {
        Res<W> zero;
#pragma unroll
        for (int i = 0; i < W; i++) zero.v[i] = 0;
        acc = sub(zero, acc, M);
    }
    return acc;
}
/* }}} */

/* {{{ the groups: Lucas sequences V_k(X) for P+-1, Montgomery curves */
template<int W> struct Lucas {
    typedef Res<W> Elt;
    Mod<W> const & M;
    Res<W> two;
    __device__ Lucas(Mod<W> const & M) : M(M) { two = add(M.one, M.one, M); }
    /* V_{2k} = V_k^2 - 2 */
    __device__ __forceinline__ Elt dbl(Elt const & a) const { return sub(sqr(a, M), two, M); }
    /* V_{m+n} = V_m V_n - V_{m-n} */
    __device__ __forceinline__ Elt dadd(Elt const & a, Elt const & b, Elt const & d) const
    { return sub(mul(a, b, M), d, M); }
};

template<int W> struct MPt { Res<W> X, Z; };

template<int W> struct Monty {
    typedef MPt<W> Elt;
    Mod<W> const & M;
    Res<W> b;   /* (A+2)/4 */
    __device__ Monty(Mod<W> const & M, Res<W> const & b) : M(M), b(b) {}
    __device__ __forceinline__ Elt dbl(Elt const & P) const
    {
        Res<W> u = sqr(add(P.X, P.Z, M), M);
        Res<W> v = sqr(sub(P.X, P.Z, M), M);
        Res<W> t = sub(u, v, M);        /* 4XZ */
        Elt R;
        R.X = mul(u, v, M);
        R.Z = mul(t, add(v, mul(b, t, M), M), M);
        return R;
    }
    /* P + Q given D = P - Q */
    __device__ __forceinline__ Elt dadd(Elt const & P, Elt const & Q, Elt const & D) const
    {
        Res<W> u = mul(sub(P.X, P.Z, M), add(Q.X, Q.Z, M), M);
        Res<W> v = mul(add(P.X, P.Z, M), sub(Q.X, Q.Z, M), M);
        Res<W> s = sqr(add(u, v, M), M);
        Res<W> d = sqr(sub(u, v, M), M);
        Elt R;
        R.X = mul(D.Z, s, M);
        R.Z = mul(D.X, d, M);
        return R;
    }
};

/* PRAC: P <- k*P, k the scalar the bytecode encodes. Same rules as
 * bytecode_prac_interpret_montgomery_internal (ecm.cpp). */
template<class G>
__device__ typename G::Elt prac(G const & g, typename G::Elt const & P,
        const uint8_t * bc)
{
    typedef typename G::Elt E;
    E R0 = P, R1, R2, R3, R4;
    for (int i = 0;; i++) {
        uint8_t const op = __ldg(bc + i);
        switch (op) {
            case PRAC_SWAP: { E t = R0; R0 = R1; R1 = t; } break;
            case PRAC_SUBBLOCK_INIT: R1 = R0; R2 = R0; R0 = g.dbl(R0); break;
            case PRAC_SUBBLOCK_FINAL: R0 = g.dadd(R0, R1, R2); break;
            case PRAC_BLOCK_FINAL: R1 = g.dadd(R0, R1, R2); return R1;
            case 1:
                R3 = g.dadd(R0, R1, R2);
                R4 = g.dadd(R3, R0, R1);
                R1 = g.dadd(R1, R3, R0);
                R0 = R4;
                break;
            case 2: R1 = g.dadd(R0, R1, R2); R0 = g.dbl(R0); break;
            case 3: R2 = g.dadd(R1, R0, R2); { E t = R1; R1 = R2; R2 = t; } break;
            case 4: R1 = g.dadd(R1, R0, R2); R0 = g.dbl(R0); break;
            case 5: R2 = g.dadd(R2, R0, R1); R0 = g.dbl(R0); break;
            case 6:
                R3 = g.dbl(R0);
                R4 = g.dadd(R0, R1, R2);
                R0 = g.dadd(R3, R0, R0);
                R2 = g.dadd(R3, R4, R2);
                { E t = R1; R1 = R2; R2 = t; }
                break;
            case 7:
                R3 = g.dadd(R0, R1, R2);
                R1 = g.dadd(R3, R0, R1);
                R3 = g.dbl(R0);
                R0 = g.dadd(R0, R3, R0);
                break;
            case 8:
                R3 = g.dadd(R0, R1, R2);
                R2 = g.dadd(R2, R0, R1);
                { E t = R1; R1 = R3; R3 = t; }
                R3 = g.dbl(R0);
                R0 = g.dadd(R0, R3, R0);
                break;
            case 9: R2 = g.dadd(R2, R1, R0); R1 = g.dbl(R1); break;
            case 10: R1 = g.dadd(R0, R1, R2); R2 = R1; R0 = g.dbl(R1); break;
            case 11:
                R2 = g.dadd(R1, R0, R2);
                /* (R1, R2, R0) := (R0, R1, R2) */
                { E t = R1; R1 = R2; R2 = t; }
                { E t = R0; R0 = R1; R1 = t; }
                break;
            case 12:
                R3 = g.dadd(R1, R0, R2);
                R2 = g.dadd(R0, R3, R1);
                R1 = R2;
                R0 = g.dbl(R2);
                break;
            case 13:
                R3 = R1;
                R1 = g.dadd(R1, R0, R2);
                R2 = R0;
                R0 = g.dadd(R0, R1, R3);
                break;
            default:
                /* unknown code: cannot happen with CADO's encoder */
                return R1;
        }
    }
}

/* (k*P, (k+1)*P) for k >= 1, Montgomery ladder */
template<class G>
__device__ void ladder(G const & g, typename G::Elt const & P, uint32_t k,
        typename G::Elt & Rk, typename G::Elt & Rk1)
{
    typedef typename G::Elt E;
    E R0 = P, R1 = g.dbl(P);
    int const top = 31 - __clz(k);
    for (int i = top - 1; i >= 0; i--) {
        if ((k >> i) & 1) {
            R0 = g.dadd(R1, R0, P);
            R1 = g.dbl(R1);
        } else {
            R1 = g.dadd(R1, R0, P);
            R0 = g.dbl(R0);
        }
    }
    Rk = R0;
    Rk1 = R1;
}

/* The baby steps u*P (u in U) and w*P, as in pp1_stage2 / ecm_stage2:
 * two progressions 1+6k and 5+6k. */
template<class G>
__device__ void baby_steps(G const & g, typename G::Elt const & P,
        const uint32_t * U, uint32_t U_len, uint32_t w,
        typename G::Elt * uP, typename G::Elt & wP)
{
    typedef typename G::Elt E;
    E ap1_0 = P;
    E P2 = g.dbl(P);
    E P6 = g.dadd(P2, P, P);        /* 3P */
    E ap5_0 = g.dadd(P6, P2, P);    /* 5P */
    P6 = g.dbl(P6);                 /* 6P */
    E ap1_1 = g.dadd(P6, P, ap5_0); /* 7P */
    E ap5_1 = g.dadd(P6, ap5_0, P); /* 11P */

    uint32_t k = 0;
    if (k < U_len && __ldg(U + k) == 1) uP[k++] = ap1_0;
    if (k < U_len && __ldg(U + k) == 5) uP[k++] = ap5_0;
    uint32_t u1 = 7, u5 = 11;
    while (k < U_len) {
        uint32_t const u = __ldg(U + k);
        if (u == u1) { uP[k++] = ap1_1; continue; }
        if (u == u5) { uP[k++] = ap5_1; continue; }
        E t = g.dadd(ap1_1, P6, ap1_0);
        ap1_0 = ap1_1; ap1_1 = t; u1 += 6;
        t = g.dadd(ap5_1, P6, ap5_0);
        ap5_0 = ap5_1; ap5_1 = t; u5 += 6;
    }
    if (w == 6) {
        wP = P6;
    } else if (w == 12) {
        wP = g.dbl(P6);
    } else if (w % 12 == 0) {
        E t = g.dadd(ap1_1, P6, ap1_0);
        wP = g.dadd(t, ap5_1, P2);
    } else {
        P2 = g.dbl(P2);     /* 4P */
        wP = g.dadd(ap1_1, ap5_1, P2);
    }
}
/* }}} */

struct blob_ptrs {
    const fgpu_chain * chains;
    const fgpu_method * methods;
    const uint64_t * p64;
    const uint32_t * p32;
    const uint16_t * p16;
    const uint8_t * p8;
};

/* {{{ stage 2 */
/* P+-1: r = prod over the pairs of (V_{vw}(X) - V_u(X)). Same values as
 * pp1_stage2 (Lucas sequence values do not depend on the chain). */
template<int W>
__device__ int stage2_pp1(Res<W> & r, Res<W> const & X, fgpu_method const & m,
        blob_ptrs const & B, Mod<W> const & M, Res<W> * Xu)
{
    Lucas<W> g(M);
    Res<W> Xw;
    baby_steps(g, X, B.p32 + m.U_off, m.U_len, m.w, Xu, Xw);

    Res<W> V0, V1;
    if (m.vmin == 0) {
        V0 = g.two;
        V1 = Xw;
    } else {
        ladder(g, Xw, m.vmin, V0, V1);
    }

    int bt = 0;
    r = M.one;
    const uint16_t * pp = B.p16 + m.pairs_off;
    for (uint32_t v = m.vmin;; v++) {
        Res<W> const r_bak = r;
        uint16_t u;
        for (; (u = __ldg(pp)) < FGPU_PAIR_INCR_V; pp++)
            r = mul(r, sub(V0, Xu[u], M), M);
        if (is0(r)) {
            r = r_bak;
            bt = 1;
            break;
        }
        if (u == FGPU_PAIR_END) break;
        pp++;
        /* next giant step */
        Res<W> const V2 = g.dadd(V1, Xw, V0);
        V0 = V1;
        V1 = V2;
    }
    return bt;
}

/* ECM: r = prod over the pairs of (x_{vwP} - x_{uP}), with the baby
 * points normalized (Z = 1) and each giant point's Z multiplied in
 * (ecm_stage2 brings all points to a common Z, whose product is in r). */
template<int W>
__device__ int stage2_ecm(Res<W> & r, MPt<W> const & P, Res<W> const & b,
        fgpu_method const & m, blob_ptrs const & B, Mod<W> const & M,
        MPt<W> * uP)
{
    Monty<W> g(M, b);
    MPt<W> wP;
    uint32_t const U_len = m.U_len;
    baby_steps(g, P, B.p32 + m.U_off, U_len, m.w, uP, wP);

    /* normalize the baby points: uP[k].X <- X_k / Z_k. Forward pass:
     * X_k <- X_k * prod_{j<k} Z_j; backward pass with S = prod_{j>k} Z_j:
     * x_k = X_k * S / prod Z. */
    Res<W> pre = M.one;
    for (uint32_t k = 0; k < U_len; k++) {
        uP[k].X = mul(uP[k].X, pre, M);
        pre = mul(pre, uP[k].Z, M);
    }
    Res<W> ipre;
    if (!inv(ipre, pre, M)) {
        /* a Z is 0 mod some prime: that prime divides prod Z, as it
         * divides the common Z of ecm_stage2 */
        r = pre;
        return 0;
    }
    Res<W> S = ipre;
    for (int k = (int) U_len - 1; k >= 0; k--) {
        uP[k].X = mul(uP[k].X, S, M);
        S = mul(S, uP[k].Z, M);
    }

    MPt<W> V0, V1;
    bool inf0 = false;
    if (m.vmin == 0) {
        inf0 = true;    /* 0*wP, the point at infinity */
        V1 = wP;
    } else {
        ladder(g, wP, m.vmin, V0, V1);
    }

    int bt = 0;
    r = M.one;
    const uint16_t * pp = B.p16 + m.pairs_off;
    for (uint32_t v = m.vmin;; v++) {
        Res<W> const r_bak = r;
        uint16_t u;
        if (inf0) {
            /* ecm_stage2 takes X(0*wP) = 0: the term is -x_u */
            Res<W> zero;
#pragma unroll
            for (int i = 0; i < W; i++) zero.v[i] = 0;
            for (; (u = __ldg(pp)) < FGPU_PAIR_INCR_V; pp++)
                r = mul(r, sub(zero, uP[u].X, M), M);
        } else {
            r = mul(r, V0.Z, M);
            for (; (u = __ldg(pp)) < FGPU_PAIR_INCR_V; pp++)
                r = mul(r, sub(V0.X, mul(uP[u].X, V0.Z, M), M), M);
        }
        if (is0(r)) {
            r = r_bak;
            bt = 1;
            break;
        }
        if (u == FGPU_PAIR_END) break;
        pp++;
        /* next giant step; 2*wP is a doubling (no difference point) */
        MPt<W> V2;
        if (inf0) {
            V2 = g.dbl(V1);
            inf0 = false;
        } else {
            V2 = g.dadd(V1, wP, V0);
        }
        V0 = V1;
        V1 = V2;
    }
    return bt;
}
/* }}} */

/* {{{ the methods. Each returns the factor found (1 if none) */
template<int W>
__device__ u128 run_pm1(fgpu_method const & m, blob_ptrs const & B,
        Mod<W> const & M, Res<W> * scratch)
{
    /* x = 2^E, left to right */
    Res<W> x = M.one;
    int started = 0;
    for (int i = (int) m.E_len - 1; i >= 0; i--) {
        uint64_t const e = __ldg(B.p64 + m.E_off + i);
        for (int j = 63; j >= 0; j--) {
            if (started) x = sqr(x, M);
            if ((e >> j) & 1) {
                x = add(x, x, M);
                started = 1;
            }
        }
    }
    /* the 2s, with backtracking */
    Res<W> t = x;
    for (uint32_t i = 0; i < m.exp2; i++) {
        x = sqr(x, M);
        if (eq(x, M.one)) {
            x = t;
            break;
        }
        t = x;
    }
    u128 f = res_gcd(sub(x, M.one, M), M);
    if (f > 1 || m.w == 0 || m.B1 >= m.B2) return f;

    Res<W> X;
    if (!inv(X, x, M)) return res_gcd(x, M);   /* cannot happen, n odd */
    X = add(X, x, M);
    Res<W> r;
    stage2_pp1(r, X, m, B, M, scratch);
    return res_gcd(r, M);
}

template<int W>
__device__ u128 run_pp1(fgpu_method const & m, blob_ptrs const & B,
        Mod<W> const & M, Res<W> * scratch)
{
    Lucas<W> g(M);
    Res<W> d = bigint_mod(m.c[1], B.p64, M), b;
    if (!inv(b, d, M)) return res_gcd(d, M);
    b = mul(b, bigint_mod(m.c[0], B.p64, M), M);

    if (m.bc_len) b = prac(g, b, B.p8 + m.bc_off);
    Res<W> t = b;
    int bt = 0;
    for (uint32_t i = 0; i < m.exp2; i++) {
        b = g.dbl(b);
        if (eq(b, g.two)) {
            b = t;
            bt = 1;
            break;
        }
        t = b;
    }
    u128 f = res_gcd(sub(b, g.two, M), M);
    if (bt == 0 && f == 1 && m.w != 0 && m.B1 < m.B2) {
        Res<W> r;
        stage2_pp1(r, b, m, B, M, scratch);
        f = res_gcd(r, M);
    }
    return f;
}

template<int W>
__device__ u128 run_ecm(fgpu_method const & m, blob_ptrs const & B,
        Mod<W> const & M, MPt<W> * scratch)
{
    Res<W> d = bigint_mod(m.c[1], B.p64, M), b;
    if (!inv(b, d, M)) return res_gcd(d, M);
    b = mul(b, bigint_mod(m.c[0], B.p64, M), M);
    MPt<W> P;
    P.X = bigint_mod(m.c[2], B.p64, M);
    P.Z = bigint_mod(m.c[3], B.p64, M);

    Monty<W> g(M, b);
    if (m.bc_len) P = prac(g, P, B.p8 + m.bc_off);
    MPt<W> Pt = P;
    int bt = 0;
    for (uint32_t i = 0; i < m.exp2; i++) {
        P = g.dbl(P);
        if (is0(P.Z)) {
            P = Pt;
            bt = 1;
            break;
        }
        Pt = P;
    }
    u128 f = res_gcd(P.Z, M);
    if (bt == 0 && f == 1 && m.w != 0 && m.B1 < m.B2) {
        Res<W> r;
        stage2_ecm(r, P, b, m, B, M, scratch);
        f = res_gcd(r, M);
    }
    return f;
}
/* }}} */

/* {{{ primality: arithxx is_prime (base 2 strong test, then the strong
 * Lucas V test with Q = 1 and the first P >= 3 with (P^2-4 / n) = -1) */
template<int W>
__device__ bool sprp2(Mod<W> const & M, u128 n)
{
    u128 const nm1 = n - 1;
    int const s = u128_ctz(nm1);
    u128 const d = nm1 >> s;
    Res<W> x = M.one, two = add(M.one, M.one, M);
    for (int i = u128_bits(d) - 1; i >= 0; i--) {
        x = sqr(x, M);
        if ((d >> i) & 1) x = add(x, x, M);
    }
    (void) two;
    Res<W> zero;
#pragma unroll
    for (int i = 0; i < W; i++) zero.v[i] = 0;
    Res<W> const mone = sub(zero, M.one, M);
    if (eq(x, M.one) || eq(x, mone)) return true;
    for (int i = 1; i < s; i++) {
        x = sqr(x, M);
        if (eq(x, mone)) return true;
    }
    return false;
}

template<int W>
__device__ bool lucas_sprp(Mod<W> const & M, u128 n)
{
    if (!(n & 1)) return n == 2;
    uint64_t P = 3;
    for (int i = 0;; i++) {
        /* D = P^2 - 4 */
        u128 const D = (u128) P * P - 4;
        if (u128_jacobi(D, n) == -1) break;
        P += 2;     /* as arithxx: P = 3, 5, 7, ... */
        if (i == 20 && u128_is_square(n)) return false;
    }
    u128 ell = n + 1;
    int const k = u128_ctz(ell);
    ell >>= k;
    Lucas<W> g(M);
    Res<W> const X = small<W>(P, M);
    /* v = V_ell(X), ladder over the bits of ell */
    Res<W> v0 = X, v1 = g.dbl(X);
    for (int i = u128_bits(ell) - 2; i >= 0; i--) {
        if ((ell >> i) & 1) {
            v0 = g.dadd(v1, v0, X);
            v1 = g.dbl(v1);
        } else {
            v1 = g.dadd(v1, v0, X);
            v0 = g.dbl(v0);
        }
    }
    Res<W> v = v0;
    Res<W> zero;
#pragma unroll
    for (int i = 0; i < W; i++) zero.v[i] = 0;
    Res<W> const mtwo = sub(zero, g.two, M);
    if (eq(v, g.two) || eq(v, mtwo)) return true;
    for (int i = 0; i < k; i++) {
        if (is0(v)) return true;
        v = g.dbl(v);
        if (eq(v, g.two) || eq(v, mtwo)) return false;
    }
    return eq(v, g.two);
}

template<int W>
__device__ bool is_prime(u128 n)
{
    if (n < 2) return false;
    if (!(n & 1)) return n == 2;
    if (n < 9) return true;     /* 3, 5, 7 */
    Mod<W> M;
    mod_init(M, n);
    if (!sprp2(M, n)) return false;
    if (n < 2047) return true;  /* arithxx_api64::sprp2_is_enough */
    return lucas_sprp(M, n);
}
/* }}} */

/* {{{ one side: facul_all for that side, then facul_aux */
struct side_state {
    u128 todo[FGPU_MAXC];
    int ntodo;
    int nprimes;
};

/* facul_doit_onefm after the method has run: returns FGPU_NOT_SMOOTH,
 * FGPU_MAYBE (with 0, 1 or 2 composites added to next), FGPU_SMOOTH, or
 * FGPU_CPU on overflow. */
template<int W>
__device__ int handle_factor(u128 n, u128 f, fgpu_chain const & ch,
        fgpu_result * out, int & nprimes, u128 * next, int & nnext,
        bool & found)
{
    found = false;
    if (f == 1 || f == n) {
        next[nnext++] = n;
        return FGPU_MAYBE;
    }
    found = true;
    unsigned const lpb = ch.lpb;
    double const f_dbl = u128_to_double(f);
    bool fprime = f_dbl < ch.BB;
    if (fprime && (unsigned) u128_bits(f) > lpb) return FGPU_NOT_SMOOTH;
    else if (2 * lpb < (unsigned) u128_bits(f) && f_dbl < ch.BBB) return FGPU_NOT_SMOOTH;

    u128 const c = n / f;
    double const c_dbl = u128_to_double(c);
    bool cprime = c_dbl < ch.BB;
    if (cprime && (unsigned) u128_bits(c) > lpb) return FGPU_NOT_SMOOTH;
    else if (2 * lpb < (unsigned) u128_bits(c) && c_dbl < ch.BBB) return FGPU_NOT_SMOOTH;

    if (!fprime) {
        fprime = is_prime<W>(f);
        if (fprime && (unsigned) u128_bits(f) > lpb) return FGPU_NOT_SMOOTH;
    }
    if (!cprime) {
        cprime = is_prime<W>(c);
        if (cprime && (unsigned) u128_bits(c) > lpb) return FGPU_NOT_SMOOTH;
    }
    if (nprimes + 2 > FGPU_MAXP || nnext + 2 > FGPU_MAXC) return FGPU_CPU;
    if (fprime) {
        out->p[nprimes][0] = (uint64_t) f;
        out->p[nprimes][1] = (uint64_t)(f >> 64);
        nprimes++;
    } else {
        next[nnext++] = f;
    }
    if (cprime) {
        out->p[nprimes][0] = (uint64_t) c;
        out->p[nprimes][1] = (uint64_t)(c >> 64);
        nprimes++;
    } else {
        next[nnext++] = c;
    }
    return FGPU_MAYBE;
}

template<int W>
__device__ u128 run_method(fgpu_method const & m, blob_ptrs const & B,
        Mod<W> const & M, MPt<W> * scratch)
{
    switch (m.type) {
        case FGPU_PM1: return run_pm1<W>(m, B, M, (Res<W> *) scratch);
        case FGPU_PP1: return run_pp1<W>(m, B, M, (Res<W> *) scratch);
        case FGPU_ECM: return run_ecm<W>(m, B, M, scratch);
    }
    return 1;
}

template<int W>
__device__ int run_side(u128 n, fgpu_chain const & ch, blob_ptrs const & B,
        fgpu_result * out, MPt<W> * scratch)
{
    u128 todo[FGPU_MAXC];
    int ntodo = 1;
    todo[0] = n;
    int nprimes = 0;
    bool split = false;
    Mod<W> M0;
    mod_init(M0, n);    /* the single composite of the unsplit phase */

    for (uint32_t i = 0; i < ch.nm && ntodo; i++) {
        fgpu_method const m = B.methods[ch.m0 + i];
        u128 next[FGPU_MAXC];
        int nnext = 0;
        bool any = false;
        for (int t = 0; t < ntodo; t++) {
            u128 const c = todo[t];
            u128 f;
            if (!split) {
                f = run_method<W>(m, B, M0, scratch);
            } else {
                Mod<W> M;
                mod_init(M, c);
                f = run_method<W>(m, B, M, scratch);
            }
            bool found;
            int const st = handle_factor<W>(c, f, ch, out, nprimes, next, nnext, found);
            if (st == FGPU_NOT_SMOOTH || st == FGPU_CPU) {
                out->nprimes = nprimes;
                return st;
            }
            any |= found;
        }
        if (!split && !any && (m.flags & (FGPU_FLAG_STOP_UNSPLIT | FGPU_FLAG_LAST))) {
            out->nprimes = nprimes;
            return FGPU_NOT_SMOOTH;
        }
        split |= any;
        for (int t = 0; t < nnext; t++) todo[t] = next[t];
        ntodo = nnext;
    }
    out->nprimes = nprimes;
    return ntodo == 0 ? FGPU_SMOOTH : FGPU_MAYBE;
}

template<int W>
__global__ void __launch_bounds__(128)
facul_kernel(const fgpu_job * jobs, fgpu_result * res, unsigned int njobs,
        const uint8_t * blob, MPt<W> * scratch_all)
{
    unsigned int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= njobs) return;
    fgpu_blob_header const * h = (fgpu_blob_header const *) blob;
    blob_ptrs B;
    B.chains = (const fgpu_chain *) (blob + h->off_chains);
    B.methods = (const fgpu_method *) (blob + h->off_methods);
    B.p64 = (const uint64_t *) (blob + h->off64);
    B.p32 = (const uint32_t *) (blob + h->off32);
    B.p16 = (const uint16_t *) (blob + h->off16);
    B.p8 = blob + h->off8;

    fgpu_job const j = jobs[i];
    fgpu_chain const ch = B.chains[j.chain];
    u128 const n = ((u128) j.n[1] << 64) | j.n[0];
    fgpu_result * out = res + i;
    /* the stage 2 baby step table of this thread, interleaved so that a
     * warp's accesses to entry k are contiguous */
    (void) scratch_all;
    MPt<W> scratch[FGPU_UMAX];
    out->status = run_side<W>(n, ch, B, out, scratch);
}
/* }}} */

/* {{{ host side */
struct fgpu_ctx {
    int device;
    cudaStream_t stream;
    uint8_t * d_blob = nullptr;
    size_t blob_alloc = 0;
    fgpu_job * d_jobs = nullptr;
    fgpu_result * d_res = nullptr;
    fgpu_job * h_jobs = nullptr;
    fgpu_result * h_res = nullptr;
    size_t cap = 0;
    double kernel_time = 0;
    unsigned long njobs = 0;
};

#define CK(x) do {                                                      \
    cudaError_t const e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                            \
        fprintf(stderr, "facul-gpu: %s failed: %s\n", #x,               \
                cudaGetErrorString(e_));                                \
        abort();                                                        \
    }                                                                   \
} while (0)

int fgpu_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

struct fgpu_ctx * fgpu_create(int device)
{
    int ndev = fgpu_device_count();
    if (ndev <= 0) return nullptr;
    if (device < 0) {
        const char * e = getenv("CADO_LAS_GPU_DEVICE");
        if (e) device = atoi(e);
        else {
            const char * lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
            if (!lr) lr = getenv("MPI_LOCALRANKID");
            if (!lr) lr = getenv("SLURM_LOCALID");
            device = lr ? atoi(lr) : cado_gpu_claim("las", ndev);
        }
    }
    device %= ndev;
    CK(cudaSetDevice(device));
    if (!cado_cuda_can_run("las GPU cofactorization")) return nullptr;
    auto * c = new fgpu_ctx;
    c->device = device;
    /* the batch thread waits for the GPU: let it sleep (fails harmlessly
     * if the context exists already) */
    if (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess)
        cudaGetLastError();
    CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    return c;
}

void fgpu_free(struct fgpu_ctx * c)
{
    if (!c) return;
    cudaSetDevice(c->device);
    cudaFree(c->d_blob);
    cudaFree(c->d_jobs);
    cudaFree(c->d_res);
    cudaFreeHost(c->h_jobs);
    cudaFreeHost(c->h_res);
    cudaStreamDestroy(c->stream);
    delete c;
}

int fgpu_device(const struct fgpu_ctx * c) { return c->device; }

void fgpu_set_blob(struct fgpu_ctx * c, const void * blob, size_t bytes)
{
    CK(cudaSetDevice(c->device));
    if (bytes > c->blob_alloc) {
        CK(cudaStreamSynchronize(c->stream));
        cudaFree(c->d_blob);
        c->blob_alloc = bytes + bytes / 2;
        CK(cudaMalloc(&c->d_blob, c->blob_alloc));
    }
    CK(cudaMemcpyAsync(c->d_blob, blob, bytes, cudaMemcpyHostToDevice, c->stream));
    CK(cudaStreamSynchronize(c->stream));
}

static void ensure_cap(struct fgpu_ctx * c, size_t n)
{
    if (n <= c->cap) return;
    CK(cudaStreamSynchronize(c->stream));
    cudaFree(c->d_jobs);
    cudaFree(c->d_res);
    cudaFreeHost(c->h_jobs);
    cudaFreeHost(c->h_res);
    c->cap = n;
    CK(cudaMalloc(&c->d_jobs, n * sizeof(fgpu_job)));
    CK(cudaMalloc(&c->d_res, n * sizeof(fgpu_result)));
    CK(cudaMallocHost(&c->h_jobs, n * sizeof(fgpu_job)));
    CK(cudaMallocHost(&c->h_res, n * sizeof(fgpu_result)));
}

void fgpu_run(struct fgpu_ctx * c, const struct fgpu_job * jobs,
        struct fgpu_result * res, size_t n, int words)
{
    if (n == 0) return;
    CK(cudaSetDevice(c->device));
    ensure_cap(c, n);
    memcpy(c->h_jobs, jobs, n * sizeof(fgpu_job));
    CK(cudaMemcpyAsync(c->d_jobs, c->h_jobs, n * sizeof(fgpu_job),
                cudaMemcpyHostToDevice, c->stream));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, c->stream));
    unsigned const tpb = 128;
    unsigned const nb = (unsigned) ((n + tpb - 1) / tpb);
    if (words == 1)
        facul_kernel<1><<<nb, tpb, 0, c->stream>>>(c->d_jobs, c->d_res,
                (unsigned) n, c->d_blob, nullptr);
    else
        facul_kernel<2><<<nb, tpb, 0, c->stream>>>(c->d_jobs, c->d_res,
                (unsigned) n, c->d_blob, nullptr);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1, c->stream));
    CK(cudaMemcpyAsync(c->h_res, c->d_res, n * sizeof(fgpu_result),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    c->kernel_time += ms * 1e-3;
    c->njobs += n;
    memcpy(res, c->h_res, n * sizeof(fgpu_result));
}

void fgpu_stats(const struct fgpu_ctx * c, double * kernel_seconds,
        unsigned long * njobs)
{
    *kernel_seconds = c->kernel_time;
    *njobs = c->njobs;
}
/* }}} */

/* vim: set sw=4 fdm=marker: */
