/* Large integer multiplication on the GPU; see sqrt_gpu_mul.h. */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"

#include "sqrt_gpu_mul.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {           \
    fprintf(stderr, "sqrt GPU: %s failed at %s:%d: %s\n", #x,               \
            __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)

#define NP 3            /* primes */
#define LOG2_MAX 40     /* the primes are c 2^40 + 1 */

struct prime_t {
    uint64_t p, minv, r2;       /* modulus, -p^-1 mod 2^64, 2^128 mod p */
    uint64_t w0;                /* a root of unity of order 2^40 */
};

/* ---- host arithmetic ---- */

static uint64_t mulmod_h(uint64_t a, uint64_t b, uint64_t m)
{
    return (uint64_t) ((unsigned __int128) a * b % m);
}

static uint64_t powmod_h(uint64_t a, uint64_t e, uint64_t m)
{
    uint64_t r = 1;
    for ( ; e ; e >>= 1, a = mulmod_h(a, a, m))
        if (e & 1) r = mulmod_h(r, a, m);
    return r;
}

/* deterministic for n < 3.3e24 */
static bool is_prime_h(uint64_t n)
{
    if (n < 2) return false;
    static const uint64_t B[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    for (uint64_t b : B) if (n % b == 0) return n == b;
    uint64_t d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    for (uint64_t b : B) {
        uint64_t x = powmod_h(b, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int i = 1; i < s && comp; i++) {
            x = mulmod_h(x, x, n);
            if (x == n - 1) comp = false;
        }
        if (comp) return false;
    }
    return true;
}

/* ---- device arithmetic: Montgomery, m odd, m < 2^63 ---- */

__device__ __forceinline__ uint64_t redc(uint64_t hi, uint64_t lo, uint64_t m, uint64_t minv)
{
    const uint64_t q = lo * minv;
    const uint64_t r = hi + __umul64hi(q, m) + (lo != 0);
    return r >= m ? r - m : r;
}
__device__ __forceinline__ uint64_t mulredc(uint64_t a, uint64_t b, uint64_t m, uint64_t minv)
{
    return redc(__umul64hi(a, b), a * b, m, minv);
}
__device__ __forceinline__ uint64_t addm(uint64_t a, uint64_t b, uint64_t m)
{
    uint64_t r = a + b;
    return r >= m ? r - m : r;
}
__device__ __forceinline__ uint64_t subm(uint64_t a, uint64_t b, uint64_t m)
{
    return a >= b ? a - b : a + m - b;
}

__constant__ prime_t c_pr[NP];

/* the operand, reduced mod each prime, in Montgomery form, padded to N */
__global__ void load_kernel(uint64_t * A, const uint64_t * a, size_t n, size_t N)
{
    const int j = blockIdx.y;
    const prime_t P = c_pr[j];
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += (size_t) gridDim.x * blockDim.x) {
        const uint64_t x = i < n ? a[i] % P.p : 0;
        A[j * N + i] = mulredc(x, P.r2, P.p, P.minv);
    }
}

/* T[k] = w^k (Montgomery), k < N/2, w of order N (or its inverse) */
__global__ void twiddle_kernel(uint64_t * T, size_t half, const uint64_t * wR)
{
    const int j = blockIdx.y;
    const prime_t P = c_pr[j];
    for (size_t k = (size_t) blockIdx.x * blockDim.x + threadIdx.x; k < half;
            k += (size_t) gridDim.x * blockDim.x) {
        uint64_t r = mulredc(1, P.r2, P.p, P.minv);     /* 1 */
        uint64_t b = wR[j];
        for (size_t e = k; e; e >>= 1, b = mulredc(b, b, P.p, P.minv))
            if (e & 1) r = mulredc(r, b, P.p, P.minv);
        T[j * half + k] = r;
    }
}

/* one stage of the forward transform (decimation in frequency): blocks of
 * len = 2 h, natural order in, bit-reversed order out after all stages */
__global__ void dif_stage(uint64_t * A, size_t N, size_t h, const uint64_t * T)
{
    const int j = blockIdx.y;
    const prime_t P = c_pr[j];
    uint64_t * a = A + j * N;
    const uint64_t * t = T + j * (N / 2);
    const size_t stride = N / (2 * h);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < N / 2;
            i += (size_t) gridDim.x * blockDim.x) {
        const size_t jj = i % h, blk = i / h;
        const size_t i0 = blk * 2 * h + jj, i1 = i0 + h;
        const uint64_t u = a[i0], v = a[i1];
        a[i0] = addm(u, v, P.p);
        a[i1] = mulredc(subm(u, v, P.p), t[jj * stride], P.p, P.minv);
    }
}

/* one stage of the inverse transform (decimation in time) */
__global__ void dit_stage(uint64_t * A, size_t N, size_t h, const uint64_t * T)
{
    const int j = blockIdx.y;
    const prime_t P = c_pr[j];
    uint64_t * a = A + j * N;
    const uint64_t * t = T + j * (N / 2);
    const size_t stride = N / (2 * h);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < N / 2;
            i += (size_t) gridDim.x * blockDim.x) {
        const size_t jj = i % h, blk = i / h;
        const size_t i0 = blk * 2 * h + jj, i1 = i0 + h;
        const uint64_t u = a[i0];
        const uint64_t v = mulredc(a[i1], t[jj * stride], P.p, P.minv);
        a[i0] = addm(u, v, P.p);
        a[i1] = subm(u, v, P.p);
    }
}

__global__ void pointwise_kernel(uint64_t * A, const uint64_t * B, size_t N)
{
    const int j = blockIdx.y;
    const prime_t P = c_pr[j];
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += (size_t) gridDim.x * blockDim.x)
        A[j * N + i] = mulredc(A[j * N + i], B[j * N + i], P.p, P.minv);
}

struct ninv_t { uint64_t v[NP]; };   /* N^-1 mod p_j (plain) */

struct crt_t {
    uint64_t p1_inv_mod_p2R;    /* (p1^-1 mod p2) * R mod p2 */
    uint64_t p1p2_inv_mod_p3R;  /* ((p1 p2)^-1 mod p3) * R mod p3 */
    uint64_t p1_mod_p3R;        /* (p1 mod p3) * R mod p3 */
    uint64_t p1p2[2];           /* p1 p2, two limbs */
};
__constant__ crt_t c_crt;

/* coefficient i of the convolution, as 3 limbs */
__global__ void crt_kernel(uint64_t * out, const uint64_t * A, size_t N, size_t nr,
        ninv_t ninv)
{
    const prime_t P1 = c_pr[0], P2 = c_pr[1], P3 = c_pr[2];
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < nr;
            i += (size_t) gridDim.x * blockDim.x) {
        /* residues: Montgomery form v R; times N^-1 and out of it */
        const uint64_t r1 = mulredc(A[i], ninv.v[0], P1.p, P1.minv);
        const uint64_t r2 = mulredc(A[N + i], ninv.v[1], P2.p, P2.minv);
        const uint64_t r3 = mulredc(A[2 * N + i], ninv.v[2], P3.p, P3.minv);
        /* Garner: x = x1 + p1 x2 + p1 p2 x3 */
        const uint64_t x1 = r1;
        const uint64_t x2 = mulredc(subm(r2, x1 % P2.p, P2.p), c_crt.p1_inv_mod_p2R,
                P2.p, P2.minv);
        /* t = (x1 + p1 x2) mod p3 */
        const uint64_t t = addm(x1 % P3.p, mulredc(x2 % P3.p, c_crt.p1_mod_p3R, P3.p, P3.minv), P3.p);
        const uint64_t x3 = mulredc(subm(r3, t, P3.p), c_crt.p1p2_inv_mod_p3R, P3.p, P3.minv);
        /* assemble: p1 x2 (2 limbs) + p1p2 x3 (3 limbs) + x1 */
        uint64_t l0 = P1.p * x2, l1 = __umul64hi(P1.p, x2), l2 = 0;
        uint64_t s;
        s = l0 + x1; l1 += s < l0; l0 = s;
        const uint64_t m0 = c_crt.p1p2[0] * x3;
        const uint64_t m1 = __umul64hi(c_crt.p1p2[0], x3) + c_crt.p1p2[1] * x3;
        const uint64_t c1 = m1 < c_crt.p1p2[1] * x3;
        const uint64_t m2 = __umul64hi(c_crt.p1p2[1], x3) + c1;
        s = l0 + m0; uint64_t c = s < l0; l0 = s;
        s = l1 + m1; uint64_t cc = s < l1; s += c; cc += s < c; l1 = s;
        l2 = m2 + cc;
        out[3 * i] = l0;
        out[3 * i + 1] = l1;
        out[3 * i + 2] = l2;
    }
}

/* ---- host ---- */

namespace {
/* shared: the primes, the twiddle tables.
 * The GPUs: every visible one (sqrt runs alone, after the sieving), each
 * thread (one dependency) on one of them in turn; one if a device is
 * given (CADO_SQRT_GPU_DEVICE, MPI or Slurm local rank).
 * CADO_GPU_SIMULATE_NDEV=n makes n logical GPUs on the physical ones
 * (testing). */
struct state_t {
    bool ok = false;
    std::vector<int> devs;      /* logical GPU -> physical device */
    prime_t pr[NP];
    /* (logical GPU, log2 N) -> fwd, inv */
    std::map<std::pair<int, int>, std::pair<uint64_t *, uint64_t *>> twiddles;
    std::atomic<unsigned> next_dev { 0 };
    std::mutex lock;            /* for twiddles and the counters */
    double seconds = 0;
    unsigned long calls = 0;
};
state_t S;

/* per thread: a stream and buffers, so that the products of several
 * threads run at the same time */
struct thread_ctx {
    bool init = false;
    int dev = 0;                /* logical GPU */
    int pdev = 0;               /* its device */
    cudaStream_t stream;
    uint64_t * d_a = nullptr; uint64_t * d_A = nullptr; uint64_t * d_B = nullptr;
    uint64_t * d_out = nullptr;
    size_t cap_N = 0, cap_in = 0;
    uint64_t * h_out = nullptr; size_t cap_hout = 0;
    uint64_t * d_w = nullptr;
    ~thread_ctx() {
        if (!init) return;
        cudaSetDevice(pdev);
        cudaFree(d_a); cudaFree(d_A); cudaFree(d_B); cudaFree(d_out);
        cudaFree(d_w); cudaFreeHost(h_out);
        cudaStreamDestroy(stream);
    }
};
thread_ctx & ctx()
{
    static thread_local thread_ctx c;
    if (!c.init) {
        c.init = true;
        c.dev = (int) (S.next_dev++ % S.devs.size());
        c.pdev = S.devs[c.dev];
        CK(cudaSetDevice(c.pdev));
        CK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        CK(cudaMalloc(&c.d_w, NP * sizeof(uint64_t)));
    }
    return c;
}
}

static uint64_t to_mont_h(uint64_t x, uint64_t p)
{
    return (uint64_t) (((unsigned __int128) (x % p) << 64) % p);
}

int sgm_init(void)
{
    static std::once_flag once;
    std::call_once(once, [] {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return;
        const char * e = getenv("CADO_SQRT_GPU_DEVICE");
        if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!e) e = getenv("SLURM_LOCALID");
        std::vector<int> cand;
        if (e) {
            cand.push_back(atoi(e) % ndev);
        } else {
            int n = ndev;
            if (const char * sim = getenv("CADO_GPU_SIMULATE_NDEV"))
                if (atoi(sim) > 0) n = atoi(sim);
            for (int k = 0; k < n; k++) cand.push_back(k % ndev);
        }
        for (int d : cand) {
            CK(cudaSetDevice(d));
            if (cado_cuda_can_run("sqrt GPU multiplication"))
                S.devs.push_back(d);
        }
        if (S.devs.empty()) return;
        /* the three largest primes c 2^40 + 1 below 2^62 */
        int k = 0;
        for (uint64_t c = (((uint64_t) 1 << 62) - 1) >> LOG2_MAX; k < NP; c--) {
            const uint64_t p = (c << LOG2_MAX) + 1;
            if (p >> 61 == 0) abort();
            if (!is_prime_h(p)) continue;
            prime_t & P = S.pr[k++];
            P.p = p;
            uint64_t inv = p;
            for (int i = 0; i < 6; i++) inv *= 2 - p * inv;
            P.minv = -inv;
            const unsigned __int128 R = ((unsigned __int128) 1 << 64) % p;
            P.r2 = (uint64_t) (R * R % p);
            /* a non-residue g: g^c has order 2^40 */
            uint64_t g = 3;
            while (powmod_h(g, (p - 1) / 2, p) == 1) g++;
            P.w0 = powmod_h(g, c, p);
        }
        /* the constants on every device used */
        crt_t C;
        const uint64_t p1 = S.pr[0].p, p2 = S.pr[1].p, p3 = S.pr[2].p;
        C.p1_inv_mod_p2R = to_mont_h(powmod_h(p1 % p2, p2 - 2, p2), p2);
        const uint64_t p1p2_mod_p3 = mulmod_h(p1 % p3, p2 % p3, p3);
        C.p1p2_inv_mod_p3R = to_mont_h(powmod_h(p1p2_mod_p3, p3 - 2, p3), p3);
        C.p1_mod_p3R = to_mont_h(p1 % p3, p3);
        const unsigned __int128 pp = (unsigned __int128) p1 * p2;
        C.p1p2[0] = (uint64_t) pp;
        C.p1p2[1] = (uint64_t) (pp >> 64);
        std::vector<int> done;
        for (int d : S.devs) {
            if (std::find(done.begin(), done.end(), d) != done.end()) continue;
            done.push_back(d);
            CK(cudaSetDevice(d));
            CK(cudaMemcpyToSymbol(c_pr, S.pr, sizeof(S.pr)));
            CK(cudaMemcpyToSymbol(c_crt, &C, sizeof(C)));
        }
        S.ok = true;
    });
    return S.ok;
}

/* the tables for N = 2^lg (made once, under the lock) */
static std::pair<const uint64_t *, const uint64_t *> twiddles(int lg)
{
    thread_ctx & c = ctx();     /* the calling thread's GPU */
    std::lock_guard<std::mutex> g(S.lock);
    auto const key = std::make_pair(c.dev, lg);
    auto it = S.twiddles.find(key);
    if (it != S.twiddles.end()) return it->second;
    const size_t N = (size_t) 1 << lg, half = N / 2;
    uint64_t * f, * iv;
    CK(cudaMalloc(&f, NP * half * sizeof(uint64_t) + 8));
    CK(cudaMalloc(&iv, NP * half * sizeof(uint64_t) + 8));
    uint64_t w[NP], wi[NP];
    for (int j = 0; j < NP; j++) {
        const uint64_t p = S.pr[j].p;
        const uint64_t x = powmod_h(S.pr[j].w0, (uint64_t) 1 << (LOG2_MAX - lg), p);
        w[j] = to_mont_h(x, p);
        wi[j] = to_mont_h(powmod_h(x, p - 2, p), p);
    }
    const dim3 grid(256, NP);
    if (half) {
        CK(cudaMemcpyAsync(c.d_w, w, sizeof(w), cudaMemcpyHostToDevice, c.stream));
        twiddle_kernel<<<grid, 256, 0, c.stream>>>(f, half, c.d_w);
        CK(cudaStreamSynchronize(c.stream));
        CK(cudaMemcpyAsync(c.d_w, wi, sizeof(wi), cudaMemcpyHostToDevice, c.stream));
        twiddle_kernel<<<grid, 256, 0, c.stream>>>(iv, half, c.d_w);
        CK(cudaStreamSynchronize(c.stream));
    }
    return S.twiddles[key] = { f, iv };
}

static double now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void transform(cudaStream_t st, uint64_t * A, const uint64_t * a, size_t n,
        int lg, const uint64_t * T)
{
    const size_t N = (size_t) 1 << lg;
    const unsigned blocks = (unsigned) std::min<size_t>(4096, (N / 2 + 255) / 256);
    const dim3 grid(blocks ? blocks : 1, NP);
    load_kernel<<<grid, 256, 0, st>>>(A, a, n, N);
    for (size_t h = N / 2; h >= 1; h /= 2)
        dif_stage<<<grid, 256, 0, st>>>(A, N, h, T);
}

void sgm_mul(uint64_t * r, const uint64_t * a, size_t na,
        const uint64_t * b, size_t nb)
{
    const double t0 = now();
    thread_ctx & c = ctx();
    CK(cudaSetDevice(c.pdev));
    const size_t nr = na + nb;
    int lg = 1;
    while (((size_t) 1 << lg) < nr) lg++;
    const size_t N = (size_t) 1 << lg;
    const auto T = twiddles(lg);
    const bool square = a == b && na == nb;

    if (N > c.cap_N) {
        cudaFree(c.d_A); cudaFree(c.d_B); cudaFree(c.d_out);
        c.cap_N = N;
        CK(cudaMalloc(&c.d_A, NP * N * sizeof(uint64_t)));
        CK(cudaMalloc(&c.d_B, NP * N * sizeof(uint64_t)));
        CK(cudaMalloc(&c.d_out, 3 * N * sizeof(uint64_t)));
    }
    if (nr > c.cap_in) {
        cudaFree(c.d_a);
        c.cap_in = nr;
        CK(cudaMalloc(&c.d_a, c.cap_in * sizeof(uint64_t)));
    }
    if (3 * nr > c.cap_hout) {
        cudaFreeHost(c.h_out);
        c.cap_hout = 3 * nr;
        CK(cudaMallocHost(&c.h_out, c.cap_hout * sizeof(uint64_t)));
    }
    ninv_t ninv;
    for (int j = 0; j < NP; j++)
        ninv.v[j] = powmod_h(N % S.pr[j].p, S.pr[j].p - 2, S.pr[j].p);

    CK(cudaMemcpyAsync(c.d_a, a, na * sizeof(uint64_t), cudaMemcpyHostToDevice, c.stream));
    if (!square)
        CK(cudaMemcpyAsync(c.d_a + na, b, nb * sizeof(uint64_t), cudaMemcpyHostToDevice, c.stream));
    transform(c.stream, c.d_A, c.d_a, na, lg, T.first);
    const unsigned blocks = (unsigned) std::min<size_t>(4096, (N + 255) / 256);
    const dim3 grid(blocks, NP);
    if (square) {
        pointwise_kernel<<<grid, 256, 0, c.stream>>>(c.d_A, c.d_A, N);
    } else {
        transform(c.stream, c.d_B, c.d_a + na, nb, lg, T.first);
        pointwise_kernel<<<grid, 256, 0, c.stream>>>(c.d_A, c.d_B, N);
    }
    for (size_t h = 1; h <= N / 2; h *= 2)
        dit_stage<<<dim3((unsigned) std::min<size_t>(4096, (N / 2 + 255) / 256), NP), 256, 0, c.stream>>>(c.d_A, N, h, T.second);
    crt_kernel<<<(unsigned) std::min<size_t>(8192, (nr + 255) / 256), 256, 0, c.stream>>>(c.d_out, c.d_A, N, nr, ninv);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(c.h_out, c.d_out, 3 * nr * sizeof(uint64_t), cudaMemcpyDeviceToHost, c.stream));
    CK(cudaStreamSynchronize(c.stream));

    /* r = sum of out[i] (3 limbs) * 2^(64 i) */
    unsigned __int128 acc = 0;
    for (size_t i = 0; i < nr; i++) {
        acc += c.h_out[3 * i];
        if (i >= 1) acc += c.h_out[3 * (i - 1) + 1];
        if (i >= 2) acc += c.h_out[3 * (i - 2) + 2];
        r[i] = (uint64_t) acc;
        acc >>= 64;
    }
    /* the product has nr limbs: nothing may be left over */
    acc += c.h_out[3 * (nr - 1) + 1];
    if (nr >= 2) acc += c.h_out[3 * (nr - 2) + 2];
    acc += (unsigned __int128) c.h_out[3 * (nr - 1) + 2] << 64;
    if (acc != 0) {
        fprintf(stderr, "sqrt GPU: product overflows %zu limbs\n", nr);
        abort();
    }
    std::lock_guard<std::mutex> g(S.lock);
    S.calls++;
    S.seconds += now() - t0;
}

int sgm_ngpu(void)
{
    return (int) S.devs.size();
}

void sgm_stats(double * seconds, unsigned long * calls)
{
    *seconds = S.seconds;
    *calls = S.calls;
}
