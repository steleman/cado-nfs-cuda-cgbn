/* The cofactor filter of the GPU siever, with CGBN: see
 * lsg_cofilter in las-sieve-gpu-kernels.h. It does on the GPU what
 * cofactoring_sync() did with GMP on the CPU for every survivor (norm,
 * division by the known primes, check_leftover_norm), and marks the
 * survivors that the CPU would reject, so that only the others go back. */

#include <cstdint>
#include <cuda_runtime.h>
#include <gmp.h>                /* before cgbn.h: CGBN's host side wants it */
#include <cgbn/cgbn.h>

#include "las-sieve-gpu-kernels.h"
#include "las-sieve-gpu-cofilter.h"

namespace {

/* {{{ the base-2 strong probable prime test of modredcul_sprp2 for n < 2^64
 * (n odd, n > 2), in 64-bit Montgomery arithmetic: the same test as the
 * CGBN one below, with one multiplication instead of 256-bit ones (the
 * CGBN version was 80% of this kernel at c140, where cofactors have at
 * most 58 bits) */
__device__ __forceinline__ uint64_t mont_mul64(uint64_t a, uint64_t b,
        uint64_t n, uint64_t ninv)
{
    /* a b R^-1 mod n, a, b < n, R = 2^64 */
    uint64_t const tlo = a * b, thi = __umul64hi(a, b);
    uint64_t const m = tlo * ninv;
    uint64_t const mlo = m * n, mhi = __umul64hi(m, n);
    /* (t + m n) / R: the low words cancel (carry iff tlo != 0) */
    uint64_t const c = tlo != 0;
    (void) mlo;
    uint64_t r = thi + mhi;
    bool const over = r < thi;
    uint64_t r2 = r + c;
    bool const over2 = r2 < r;
    if (over || over2 || r2 >= n) r2 -= n;
    return r2;
}

__device__ bool sprp2_u64(uint64_t n)
{
    uint64_t ninv = n;                  /* n^-1 mod 2^64, then its negation */
    for (int k = 0; k < 6; k++) ninv *= 2 - n * ninv;
    ninv = (uint64_t) 0 - ninv;
    uint64_t const one = ((uint64_t) 0 - n) % n;       /* R mod n */
    uint64_t const minus_one = n - one;
    uint64_t const m1 = n - 1;
    int const l = __ffsll((long long) m1) - 1;
    uint64_t const k = m1 >> l;
    int po2 = l;
    if ((n & 7) == 1 || (n & 7) == 7) po2--;   /* as mod_ul_common.c */
    /* r = 2^k, left to right, doubling by an addition */
    uint64_t r = one;
    for (int b = 63 - __clzll((long long) k); b >= 0; b--) {
        r = mont_mul64(r, r, n, ninv);
        if ((k >> b) & 1) {
            uint64_t const d = r + r;
            r = (d < r || d >= n) ? d - n : d;
        }
    }
    if (r == one || r == minus_one) return true;
    int i;
    for (i = 1; i < po2; i++) {
        r = mont_mul64(r, r, n, ninv);
        if (r == minus_one) break;
    }
    return i < po2;
}
/* }}} */

template<uint32_t BITS, uint32_t TPI>
struct cof_kernel {
    typedef cgbn_context_t<TPI> context_t;
    typedef cgbn_env_t<context_t, BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;
    typedef typename env_t::cgbn_wide_t bnw_t;

    /* cgbn_div_ui32 returns the remainder: one call a factor, and one
     * that finds no more (instead of remainder, division, remainder) */
    static __device__ void remove_all(env_t & env, bn_t & N, uint32_t p)
    {
        if (p < 2) return;
        bn_t Q;
        while (cgbn_div_ui32(env, Q, N, p) == 0)
            cgbn_set(env, N, Q);
    }

    /* mpz_get_d: truncation towards zero */
    static __device__ double get_d(env_t & env, bn_t const & N, uint32_t s)
    {
        if (s <= 53) {
            uint64_t const v = cgbn_extract_bits_ui32(env, N, 0, 32)
                | ((uint64_t) cgbn_extract_bits_ui32(env, N, 32, 21) << 32);
            return (double) v;
        }
        uint32_t const sh = s - 53;
        uint64_t const top = cgbn_extract_bits_ui32(env, N, sh, 32)
            | ((uint64_t) cgbn_extract_bits_ui32(env, N, sh + 32, 21) << 32);
        return ldexp((double) top, (int) sh);
    }

    /* modredc*_sprp2 of las: the base-2 strong probable prime test, N odd,
     * N > 2; for N of at most 64 bits (modredcul), with one squaring
     * fewer when N = +-1 mod 8, as mod_ul_common.c does */
    static __device__ bool sprp2(env_t & env, bn_t const & N, uint32_t s)
    {
        bn_t m1, k, r, two;
        bnw_t w;
        cgbn_sub_ui32(env, m1, N, 1);
        uint32_t const l = cgbn_ctz(env, m1);
        cgbn_shift_right(env, k, m1, l);
        int po2 = (int) l;
        if (s <= 64) {
            uint32_t const m8 = cgbn_get_ui32(env, N) & 7;
            if (m8 == 1 || m8 == 7) po2--;
        }
        cgbn_set_ui32(env, two, 2);
        cgbn_modular_power(env, r, two, k, N);
        if (cgbn_equals_ui32(env, r, 1) || cgbn_equals(env, r, m1))
            return true;
        int i;
        for (i = 1; i < po2; i++) {
            cgbn_sqr_wide(env, w, r);
            cgbn_rem_wide(env, r, w, N);
            if (cgbn_equals(env, r, m1))
                break;
        }
        return i < po2;
    }

    /* 0 if check_leftover_norm() rejects the survivor on this side */
    static __device__ int side_ok(env_t & env, lsg_cofilter_side const & S,
            uint32_t x, int side, int64_t i, uint32_t j,
            const uint64_t * keys, uint32_t nkeys)
    {
        if (S.deg < 0) return 1;
        /* the norm |sum c_k i^k j^(d-k)| (lognorm_base::norm) */
        uint32_t const ai = (uint32_t) (i < 0 ? -i : i);
        bn_t P, M, t, N;
        cgbn_set_ui32(env, P, 0);
        cgbn_set_ui32(env, M, 0);
        for (int k = 0; k <= S.deg; k++) {
            int sg = S.sign[k];
            if (!sg) continue;
            if (i < 0 && (k & 1)) sg = -sg;
            cgbn_load(env, t, (cgbn_mem_t<BITS> *) S.limbs[k]);
            for (int u = 0; u < k; u++) cgbn_mul_ui32(env, t, t, ai);
            for (int u = k; u < S.deg; u++) cgbn_mul_ui32(env, t, t, j);
            if (sg > 0)
                cgbn_add(env, P, P, t);
            else
                cgbn_add(env, M, M, t);
        }
        if (cgbn_compare(env, P, M) >= 0)
            cgbn_sub(env, N, P, M);
        else
            cgbn_sub(env, N, M, P);
        if (cgbn_equals_ui32(env, N, 0)) return 1;      /* the CPU's business */
        /* divide_known_primes: 2, the hits (resieved and bucket primes),
         * then the trial division primes; then q's factors */
        cgbn_shift_right(env, N, N, cgbn_ctz(env, N));
        uint64_t const want = ((uint64_t) x << 1) | (uint64_t) side;
        uint32_t lo = 0, hi = nkeys;
        while (lo < hi) {
            uint32_t const mid = (lo + hi) / 2;
            if ((keys[mid] >> 32) < want) lo = mid + 1; else hi = mid;
        }
        for (uint32_t a = lo; a < nkeys && (keys[a] >> 32) == want; a++)
            remove_all(env, N, (uint32_t) keys[a]);
        /* the td list is in groups (count, product < 2^32, primes): one
         * remainder by the product, then each prime on the 32-bit
         * remainder (p | N iff p | the remainder; dividing N by one prime
         * of the group does not change this for the others) */
        for (uint32_t a = 0; a < S.ntd; ) {
            uint32_t const cnt = S.td[a], prod = S.td[a + 1];
            uint32_t const r = cgbn_rem_ui32(env, N, prod);
            for (uint32_t b = 0; b < cnt; b++) {
                uint32_t const p = S.td[a + 2 + b];
                if (r % p == 0) remove_all(env, N, p);
            }
            a += 2 + cnt;
        }
        for (uint32_t a = 0; a < S.nqf; a++)
            remove_all(env, N, S.qf[a]);
        /* check_leftover_norm */
        uint32_t const s = BITS - cgbn_clz(env, N);
        if (s > S.mfb) return 0;
        if (S.lim == 0) return 1;
        if (s <= S.lpb) return 1;
        double const nd = get_d(env, N, s);
        double const B = (double) S.lim;
        double kB = B * B;
        for (uint32_t klpb = S.lpb; klpb < s; klpb += S.lpb, kB *= B)
            if (nd < kB) return 0;
        if (s > 126) return 1;      /* mpz_probab_prime_p there: the CPU's */
        if (s <= 64) {
            uint64_t const n = cgbn_extract_bits_ui32(env, N, 0, 32)
                | ((uint64_t) cgbn_extract_bits_ui32(env, N, 32, 32) << 32);
            return sprp2_u64(n) ? 0 : 1;
        }
        return sprp2(env, N, s) ? 0 : 1;
    }
};

template<uint32_t BITS, uint32_t TPI>
__global__ void lsg_cofilter_kernel(lsg_survivor * surv, uint32_t nsurv,
        const uint64_t * keys, uint32_t nkeys, int logI,
        const __grid_constant__ lsg_cofilter P)
{
    typedef cof_kernel<BITS, TPI> K;
    uint32_t const inst = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (inst >= nsurv) return;
    typename K::context_t ctx(cgbn_no_checks);
    typename K::env_t env(ctx.template env<typename K::env_t>());
    uint32_t const x = surv[inst].x;
    uint32_t const I = 1u << logI;
    int64_t const i = (int64_t) (x & (I - 1)) - (int64_t) (I / 2);
    uint32_t const j = x >> logI;
    bool rej = false;
    for (int side = 0; !rej && side < 2; side++)
        rej = !K::side_ok(env, P.s[side], x, side, i, j, keys, nkeys);
    if (threadIdx.x % TPI == 0)
        surv[inst].pad = rej;
}

}  /* namespace */

void lsg_cofilter_launch(cudaStream_t stream, lsg_cofilter const & P,
        lsg_survivor * surv, uint32_t nsurv, const uint64_t * keys,
        uint32_t nkeys, int logI)
{
    if (!nsurv) return;
    unsigned const threads = 128;
    if (P.bits == 256) {
        size_t const blocks = ((size_t) nsurv * 8 + threads - 1) / threads;
        lsg_cofilter_kernel<256, 8><<<(unsigned) blocks, threads, 0, stream>>>(
                surv, nsurv, keys, nkeys, logI, P);
    } else {
        size_t const blocks = ((size_t) nsurv * 16 + threads - 1) / threads;
        lsg_cofilter_kernel<512, 16><<<(unsigned) blocks, threads, 0, stream>>>(
                surv, nsurv, keys, nkeys, logI, P);
    }
}
