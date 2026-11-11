/* The character matrix on the GPU; see characters_gpu.h. */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"
#include <gmp.h>                /* before cgbn.h: CGBN's host side wants it */
#include <cgbn/cgbn.h>

#include "characters_gpu.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {           \
    fprintf(stderr, "characters GPU: %s failed at %s:%d: %s\n", #x,         \
            __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)

#define MAXSIDES 4
#define MAXDEG 8

/* ---- signs of F(a, b), exactly, with CGBN ---- */

template<uint32_t BITS, uint32_t TPI>
__global__ void sign_kernel(size_t n, const int64_t * a, const uint64_t * b,
        int deg, const cgbn_mem_t<BITS> * coef, const int8_t * csign,
        uint8_t * out)
{
    typedef cgbn_context_t<TPI> context_t;
    typedef cgbn_env_t<context_t, BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;

    const size_t inst = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (inst >= n) return;
    context_t ctx(cgbn_no_checks);
    env_t env(ctx.template env<env_t>());

    const int64_t av = a[inst];
    const uint64_t bv = b[inst];
    const uint64_t aa = av < 0 ? (uint64_t) 0 - (uint64_t) av : (uint64_t) av;

    bn_t A, B, Apow, P, Q, t, c;
    bn_t bp[MAXDEG + 1];
    /* A = |a|, B = b */
    cgbn_set_ui32(env, A, (uint32_t) (aa >> 32));
    cgbn_shift_left(env, A, A, 32);
    cgbn_add_ui32(env, A, A, (uint32_t) aa);
    cgbn_set_ui32(env, B, (uint32_t) (bv >> 32));
    cgbn_shift_left(env, B, B, 32);
    cgbn_add_ui32(env, B, B, (uint32_t) bv);
    cgbn_set_ui32(env, bp[0], 1);
    for (int j = 1; j <= deg; j++)
        cgbn_mul(env, bp[j], bp[j - 1], B);
    cgbn_set_ui32(env, Apow, 1);
    cgbn_set_ui32(env, P, 0);
    cgbn_set_ui32(env, Q, 0);
    /* F(a, b) = sum f_k a^k b^(d-k): positive terms in P, negative in Q */
    for (int k = 0; k <= deg; k++) {
        int s = csign[k];
        if (av < 0 && (k & 1)) s = -s;
        if (s != 0) {
            cgbn_load(env, c, (cgbn_mem_t<BITS> *) &coef[k]);
            cgbn_mul(env, t, c, Apow);
            cgbn_mul(env, t, t, bp[deg - k]);
            if (s > 0)
                cgbn_add(env, P, P, t);
            else
                cgbn_add(env, Q, Q, t);
        }
        if (k < deg)
            cgbn_mul(env, Apow, Apow, A);
    }
    const int cmp = cgbn_compare(env, P, Q);
    if (threadIdx.x % TPI == 0)
        out[inst] = cmp < 0;
}

/* ---- the Legendre symbols, and the 64-bit words ---- */

__device__ __forceinline__ uint64_t mulmod(uint64_t x, uint64_t y, uint64_t p)
{
    return (uint64_t) (((unsigned __int128) x * y) % p);
}

/* Jacobi symbol (x / n), n odd */
__device__ int jacobi(uint64_t x, uint64_t n)
{
    x %= n;
    int t = 1;
    while (x != 0) {
        while ((x & 1) == 0) {
            x >>= 1;
            const uint64_t r = n & 7;
            if (r == 3 || r == 5) t = -t;
        }
        const uint64_t tmp = x; x = n; n = tmp;
        if ((x & 3) == 3 && (n & 3) == 3) t = -t;
        x %= n;
    }
    return n == 1 ? t : 0;
}

struct sign_ptrs { const uint8_t * s[MAXSIDES]; };

__global__ void words_kernel(size_t n, const int64_t * a, const uint64_t * b,
        const chg_char * chars, int nchars, int ratside, sign_ptrs signs,
        uint64_t * out, uint8_t * bad)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t av = a[i];
    const uint64_t bv = b[i];
    const int nw = nchars / 64;
    bool isbad = false;
    for (int w = 0; w < nw; w++) {
        uint64_t v = 0;
        for (int k = 0; k < 64; k++) {
            const chg_char ch = chars[64 * w + k];
            uint64_t bit;
            if (ch.p == 0) {
                if (ch.r == 0)
                    bit = 0;
                else if (ch.r == 1)
                    bit = 1;
                else if (ch.r == 3)
                    bit = bv == 0;
                else if (ch.r == 2)
                    bit = signs.s[ratside][i];
                else
                    bit = signs.s[ch.r - 1024][i];
            } else {
                const uint64_t p = ch.p;
                /* b r - a (mod p), a as the CPU reduces it */
                const uint64_t ar = av < 0
                    ? (p - ((uint64_t) 0 - (uint64_t) av) % p) % p
                    : (uint64_t) av % p;
                uint64_t rr = mulmod(bv % p, ch.r % p, p);
                rr = rr >= ar ? rr - ar : rr + p - ar;
                const int j = jacobi(rr, p);
                if (j == 0) isbad = true;
                bit = j < 0;
            }
            v |= bit << k;
        }
        out[i * nw + w] = v;
    }
    bad[i] = isbad;
}

/* ---- host ---- */

static int device = -1;

int chg_available(void)
{
    if (device >= 0) return 1;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 0;
    const char * e = getenv("CADO_CHARACTERS_GPU_DEVICE");
    if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!e) e = getenv("SLURM_LOCALID");
    int const d = (e ? atoi(e) : cado_gpu_claim("characters", ndev)) % ndev;
    static int unusable = 0;
    if (unusable) return 0;
    if (cudaSetDevice(d) != cudaSuccess || !cado_cuda_can_run("characters GPU")) {
        unusable = 1;
        return 0;
    }
    device = d;
    return 1;
}

template<uint32_t BITS, uint32_t TPI>
static void run_signs(size_t n, const int64_t * d_a, const uint64_t * d_b,
        chg_poly const & P, uint8_t * d_out)
{
    std::vector<cgbn_mem_t<BITS>> coef(P.deg + 1);
    memset(coef.data(), 0, coef.size() * sizeof(coef[0]));
    for (int k = 0; k <= P.deg; k++)
        for (int l = 0; l < P.nlimbs; l++)
            coef[k]._limbs[l] = P.limbs[k * P.nlimbs + l];
    cgbn_mem_t<BITS> * d_coef;
    int8_t * d_sign;
    CK(cudaMalloc(&d_coef, coef.size() * sizeof(coef[0])));
    CK(cudaMalloc(&d_sign, P.deg + 1));
    CK(cudaMemcpy(d_coef, coef.data(), coef.size() * sizeof(coef[0]), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sign, P.sign, P.deg + 1, cudaMemcpyHostToDevice));
    const int threads = 128;
    const size_t blocks = (n * TPI + threads - 1) / threads;
    sign_kernel<BITS, TPI><<<(unsigned) blocks, threads>>>(n, d_a, d_b, P.deg, d_coef, d_sign, d_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaFree(d_coef);
    cudaFree(d_sign);
}

int chg_eval(const int64_t * a, const uint64_t * b, size_t n,
        const struct chg_char * chars, int nchars,
        const struct chg_poly * polys, int nsides,
        uint64_t * out, uint8_t * bad)
{
    if (!chg_available() || nsides > MAXSIDES || nchars % 64) return 0;
    /* which sides need signs, and the width for them */
    bool need[MAXSIDES] = { false };
    int ratside = -1;
    for (int k = 0; k < nchars; k++) {
        if (chars[k].p != 0) {
            if (chars[k].p < 3 || chars[k].p >> 62) return 0;
            continue;
        }
        if (chars[k].r == 2) {
            for (int s = 0; s < nsides; s++)
                if (polys[s].deg == 1) ratside = s;
            if (ratside < 0) return 0;
            need[ratside] = true;
        } else if (chars[k].r >= 1024) {
            if ((int) (chars[k].r - 1024) >= nsides) return 0;
            need[chars[k].r - 1024] = true;
        }
    }
    int bits = 0;
    for (int s = 0; s < nsides; s++) {
        if (!need[s]) continue;
        if (polys[s].deg < 1 || polys[s].deg > MAXDEG) return 0;
        /* |terms| < 2^(32 nlimbs + 64 deg), deg + 1 of them */
        const int b0 = 32 * polys[s].nlimbs + 64 * polys[s].deg + 5;
        if (b0 > bits) bits = b0;
    }
    if (bits > 2048) return 0;

    CK(cudaSetDevice(device));
    int64_t * d_a; uint64_t * d_b; uint64_t * d_out; uint8_t * d_bad;
    const int nw = nchars / 64;
    CK(cudaMalloc(&d_a, n * sizeof(int64_t) + 8));
    CK(cudaMalloc(&d_b, n * sizeof(uint64_t) + 8));
    CK(cudaMalloc(&d_out, n * nw * sizeof(uint64_t) + 8));
    CK(cudaMalloc(&d_bad, n + 8));
    CK(cudaMemcpy(d_a, a, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, b, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    sign_ptrs sp;
    std::vector<uint8_t *> d_signs;
    for (int s = 0; s < MAXSIDES; s++) {
        sp.s[s] = nullptr;
        if (s >= nsides || !need[s]) continue;
        uint8_t * d;
        CK(cudaMalloc(&d, n + 8));
        d_signs.push_back(d);
        sp.s[s] = d;
        if (bits <= 256)       run_signs<256, 8>(n, d_a, d_b, polys[s], d);
        else if (bits <= 512)  run_signs<512, 8>(n, d_a, d_b, polys[s], d);
        else if (bits <= 1024) run_signs<1024, 16>(n, d_a, d_b, polys[s], d);
        else                   run_signs<2048, 32>(n, d_a, d_b, polys[s], d);
    }
    chg_char * d_chars;
    CK(cudaMalloc(&d_chars, nchars * sizeof(chg_char)));
    CK(cudaMemcpy(d_chars, chars, nchars * sizeof(chg_char), cudaMemcpyHostToDevice));
    words_kernel<<<(unsigned) ((n + 127) / 128), 128>>>(n, d_a, d_b, d_chars, nchars,
            ratside < 0 ? 0 : ratside, sp, d_out, d_bad);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out, d_out, n * nw * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bad, d_bad, n, cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_bad); cudaFree(d_chars);
    for (auto d : d_signs) cudaFree(d);
    return 1;
}
