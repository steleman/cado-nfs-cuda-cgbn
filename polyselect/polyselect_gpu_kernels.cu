/* GPU collision test for polyselect's special-q stage; see
 * polyselect_gpu_kernels.h.
 *
 * For each root r (mod p^2) of the ad and each (q, rq):
 *     t = (r - rq) / q^2 mod p^2,
 *     values v = t + M + k p^2 in [0, 2M)        (v = u + M)
 * which is the set that polyselect_proots_dispatch_to_shash_flat puts
 * into its hash table. There is a collision iff two values are equal.
 *
 * Stage A (block = chunk of roots x rq): the values are partitioned by
 * their high bits (slice s = v >> w) into bins of BIN_CAP entries in
 * global memory. Stage B (block = slice x rq): the bin is bucketed by its
 * low bits in shared memory (counting sort) and each bucket is searched
 * for two equal values. A bin that overflows, or a q whose square is not
 * invertible modulo some p^2, sets the flag: the caller then decides on
 * the CPU.
 *
 * Arithmetic modulo p^2 < 2^63 (p^2 odd) is 64-bit Montgomery.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"

#include "polyselect_gpu_kernels.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {           \
    fprintf(stderr, "polyselect GPU: %s failed at %s:%d: %s\n", #x,         \
            __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)

static constexpr uint32_t BIN_CAP = 8192;   /* values per bin */
static constexpr uint32_t LOG_NB = 11;      /* buckets per bin in stage B */
static constexpr uint32_t NB = 1u << LOG_NB;
static constexpr uint32_t MAX_SLICES = 8192;
static constexpr int THREADS = 256;

struct prime_info {
    uint64_t m;         /* p^2 */
    uint64_t minv;      /* -m^-1 mod 2^64 */
    uint64_t r2;        /* 2^128 mod m */
    uint64_t um;        /* M mod m */
};

struct psg_ctx {
    int device = 0;
    cudaStream_t stream = nullptr;

    /* the ad */
    uint32_t nprimes = 0, nroots = 0;
    uint64_t twoM = 0;
    uint32_t w = 0, S = 0;      /* slice width 2^w, number of slices */
    prime_info * d_pi = nullptr;
    uint64_t * d_roots = nullptr;
    uint32_t * d_pidx = nullptr;

    /* per batch */
    size_t cap_q = 0, cap_rq = 0;
    uint64_t * d_q = nullptr;
    uint64_t * d_invR = nullptr;    /* cap_q * nprimes */
    uint8_t * d_qbad = nullptr;
    uint64_t * d_rq = nullptr;
    uint32_t * d_rqq = nullptr;
    uint32_t * d_gcnt = nullptr;    /* cap_rq * S */
    uint64_t * d_bins = nullptr;    /* cap_rq * S * BIN_CAP */
    uint8_t * d_flags = nullptr;
    unsigned int * d_overflow = nullptr;

    double seconds = 0;
    unsigned long rqs = 0, flagged = 0, overflows = 0;
};

/* ---- 64-bit Montgomery arithmetic, m odd, m < 2^63 ---- */

__device__ __forceinline__ uint64_t redc(uint64_t hi, uint64_t lo,
        uint64_t m, uint64_t minv)
{
    /* (hi 2^64 + lo) / 2^64 mod m, for hi < m */
    uint64_t mq = lo * minv;
    uint64_t uhi = __umul64hi(mq, m);
    uint64_t r = hi + uhi + (lo != 0);
    return r >= m ? r - m : r;
}

__device__ __forceinline__ uint64_t mulredc(uint64_t a, uint64_t b,
        uint64_t m, uint64_t minv)
{
    return redc(__umul64hi(a, b), a * b, m, minv);
}

/* plain inverse of a modulo m (gcd(a, m) must be 1); 0 if it is not */
__device__ uint64_t inverse_mod(uint64_t a, uint64_t m)
{
    if (a == 0) return 0;
    int64_t t = 0, nt = 1;
    uint64_t r = m, nr = a;
    while (nr) {
        uint64_t qq = r / nr;
        int64_t tt = t - (int64_t) qq * nt; t = nt; nt = tt;
        uint64_t rr = r - qq * nr; r = nr; nr = rr;
    }
    if (r != 1) return 0;
    return t < 0 ? (uint64_t) (t + (int64_t) m) : (uint64_t) t;
}

/* invR[qi * nprimes + i] = (1 / q^2 mod m_i) * 2^64 mod m_i */
__global__ void inv_kernel(const prime_info * pi, uint32_t nprimes,
        const uint64_t * qs, uint32_t nq, uint64_t * invR, uint8_t * qbad)
{
    uint64_t idx = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t) nq * nprimes) return;
    uint32_t qi = idx / nprimes, i = idx % nprimes;
    prime_info P = pi[i];
    uint64_t a = qs[qi] % P.m;
    uint64_t aR = mulredc(a, P.r2, P.m, P.minv);
    uint64_t a2 = redc(0, mulredc(aR, aR, P.m, P.minv), P.m, P.minv);
    uint64_t inv = inverse_mod(a2, P.m);
    if (inv == 0) qbad[qi] = 1;
    invR[idx] = mulredc(inv, P.r2, P.m, P.minv);
}

/* the first value v = t + M (mod m) of a root, and its modulus */
__device__ __forceinline__ uint64_t first_value(const prime_info & P,
        uint64_t root, uint64_t rqlo, uint64_t rqhi, uint64_t invR)
{
    uint64_t hi = rqhi < P.m ? rqhi : rqhi % P.m;
    uint64_t A = redc(hi, rqlo, P.m, P.minv);       /* rq / 2^64 */
    uint64_t rqm = mulredc(A, P.r2, P.m, P.minv);   /* rq mod m */
    uint64_t d = root >= rqm ? root - rqm : root + (P.m - rqm);
    uint64_t t = mulredc(d, invR, P.m, P.minv);     /* d / q^2 */
    uint64_t v = t + P.um;
    return v >= P.m ? v - P.m : v;
}

__global__ void stageA_kernel(const prime_info * pi, uint32_t nprimes,
        const uint64_t * roots, const uint32_t * pidx, uint32_t nroots,
        uint32_t chunk, const uint64_t * rq, const uint32_t * rqq,
        const uint64_t * invR, const uint8_t * qbad,
        uint64_t twoM, uint32_t w, uint32_t S,
        uint32_t * gcnt, uint64_t * bins)
{
    extern __shared__ uint32_t sh[];
    uint32_t * cnt = sh;
    uint32_t * base = sh + S;
    const uint32_t r = blockIdx.y;
    const uint32_t qi = rqq[r];
    if (qbad[qi]) return;
    const uint64_t rqlo = rq[2 * r], rqhi = rq[2 * r + 1];
    const uint64_t * inv = invR + (size_t) qi * nprimes;
    const uint32_t k0 = blockIdx.x * chunk;
    const uint32_t k1 = min(nroots, k0 + chunk);

    for (uint32_t s = threadIdx.x; s < S; s += blockDim.x) cnt[s] = 0;
    __syncthreads();
    for (uint32_t k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
        const uint32_t i = pidx[k];
        const prime_info P = pi[i];
        for (uint64_t v = first_value(P, roots[k], rqlo, rqhi, inv[i]);
                v < twoM; v += P.m)
            atomicAdd(&cnt[v >> w], 1u);
    }
    __syncthreads();
    for (uint32_t s = threadIdx.x; s < S; s += blockDim.x) {
        base[s] = cnt[s] ? atomicAdd(&gcnt[(size_t) r * S + s], cnt[s]) : 0;
        cnt[s] = 0;
    }
    __syncthreads();
    const uint64_t mask = (w >= 64) ? ~0ull : ((1ull << w) - 1);
    for (uint32_t k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
        const uint32_t i = pidx[k];
        const prime_info P = pi[i];
        for (uint64_t v = first_value(P, roots[k], rqlo, rqhi, inv[i]);
                v < twoM; v += P.m) {
            const uint32_t s = v >> w;
            const uint32_t pos = base[s] + atomicAdd(&cnt[s], 1u);
            if (pos < BIN_CAP)
                bins[((size_t) r * S + s) * BIN_CAP + pos] = v & mask;
        }
    }
}

/* exclusive scan of NB counters in place, THREADS threads */
__device__ void scan_counts(uint32_t * c, uint32_t * warp_tot)
{
    constexpr uint32_t PER = NB / THREADS;
    const uint32_t t = threadIdx.x;
    uint32_t local[PER];
    uint32_t sum = 0;
    for (uint32_t j = 0; j < PER; j++) { local[j] = c[t * PER + j]; sum += local[j]; }
    /* inclusive scan of sum over the block */
    uint32_t x = sum;
    const uint32_t lane = t & 31, wid = t >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= (uint32_t) o) x += y;
    }
    if (lane == 31) warp_tot[wid] = x;
    __syncthreads();
    if (wid == 0) {
        uint32_t y = lane < THREADS / 32 ? warp_tot[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t z = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= (uint32_t) o) y += z;
        }
        if (lane < THREADS / 32) warp_tot[lane] = y;
    }
    __syncthreads();
    uint32_t run = x - sum + (wid ? warp_tot[wid - 1] : 0);
    for (uint32_t j = 0; j < PER; j++) { c[t * PER + j] = run; run += local[j]; }
    __syncthreads();
}

__global__ void stageB_kernel(const uint32_t * rqq, const uint8_t * qbad,
        uint32_t S, const uint32_t * gcnt, const uint64_t * bins,
        uint8_t * flags, unsigned int * overflow)
{
    __shared__ uint32_t cnt[NB];
    __shared__ uint32_t data[BIN_CAP];
    __shared__ uint32_t warp_tot[THREADS / 32];
    __shared__ int found;
    const uint32_t s = blockIdx.x, r = blockIdx.y;
    if (qbad[rqq[r]]) { if (s == 0 && threadIdx.x == 0) flags[r] = 1; return; }
    const uint32_t n = gcnt[(size_t) r * S + s];
    if (n > BIN_CAP) {
        if (threadIdx.x == 0) { flags[r] = 1; atomicAdd(overflow, 1u); }
        return;
    }
    if (n < 2) return;
    const uint64_t * b = bins + ((size_t) r * S + s) * BIN_CAP;
    for (uint32_t j = threadIdx.x; j < NB; j += blockDim.x) cnt[j] = 0;
    if (threadIdx.x == 0) found = 0;
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x)
        atomicAdd(&cnt[b[j] & (NB - 1)], 1u);
    __syncthreads();
    scan_counts(cnt, warp_tot);
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        const uint64_t v = b[j];
        const uint32_t pos = atomicAdd(&cnt[v & (NB - 1)], 1u);
        data[pos] = (uint32_t) (v >> LOG_NB);
    }
    __syncthreads();
    /* bucket j is data[cnt[j-1] .. cnt[j]) */
    for (uint32_t j = threadIdx.x; j < NB; j += blockDim.x) {
        const uint32_t lo = j ? cnt[j - 1] : 0, hi = cnt[j];
        for (uint32_t x = lo; x + 1 < hi; x++)
            for (uint32_t y = x + 1; y < hi; y++)
                if (data[x] == data[y]) found = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && found) flags[r] = 1;
}

/* ---- host ---- */

struct psg_ctx * psg_ctx_create(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    auto * c = new psg_ctx;
    const char * e = getenv("CADO_POLYSELECT_GPU_DEVICE");
    if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!e) e = getenv("SLURM_LOCALID");
    /* several polyselect processes (cado-nfs.py clients) spread over
     * the GPUs */
    c->device = (e ? atoi(e) : cado_gpu_claim("polyselect", ndev)) % ndev;
    CK(cudaSetDevice(c->device));
    if (!cado_cuda_can_run("polyselect GPU collisions")) { delete c; return nullptr; }
    CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    CK(cudaMalloc(&c->d_overflow, sizeof(unsigned int)));
    return c;
}

void psg_ctx_free(struct psg_ctx * c)
{
    if (!c) return;
    cudaSetDevice(c->device);
    cudaFree(c->d_pi); cudaFree(c->d_roots); cudaFree(c->d_pidx);
    cudaFree(c->d_q); cudaFree(c->d_invR); cudaFree(c->d_qbad);
    cudaFree(c->d_rq); cudaFree(c->d_rqq); cudaFree(c->d_gcnt);
    cudaFree(c->d_bins); cudaFree(c->d_flags); cudaFree(c->d_overflow);
    cudaStreamDestroy(c->stream);
    delete c;
}

int psg_set_roots(struct psg_ctx * c, const uint32_t * primes, uint32_t nprimes,
        const uint64_t * roots, const uint32_t * root_pidx, uint32_t nroots,
        int64_t umax)
{
    if (umax <= 0 || (uint64_t) umax >= (1ull << 61)) return 0;
    std::vector<prime_info> pi(nprimes);
    for (uint32_t i = 0; i < nprimes; i++) {
        const uint64_t p = primes[i];
        if (p < 3 || p >= (1u << 31) || !(p & 1)) return 0;
        const uint64_t m = p * p;
        uint64_t inv = m;               /* m^-1 mod 2^64 by Newton */
        for (int j = 0; j < 6; j++) inv *= 2 - m * inv;
        const unsigned __int128 R = ((unsigned __int128) 1 << 64) % m;
        pi[i].m = m;
        pi[i].minv = -inv;
        pi[i].r2 = (uint64_t) ((R * R) % m);
        pi[i].um = (uint64_t) umax % m;
    }
    if (nprimes != c->nprimes) c->cap_q = 0;   /* d_invR is cap_q * nprimes */
    c->nprimes = nprimes;
    c->nroots = nroots;
    c->twoM = 2 * (uint64_t) umax;

    /* expected values per rq (an upper bound), and the slicing: about
     * half a bin per slice, slice width a power of two <= 2^43 (stage B
     * keeps v >> LOG_NB in 32 bits) */
    double n = 0;
    for (uint32_t k = 0; k < nroots; k++)
        n += (double) c->twoM / (double) pi[root_pidx[k]].m + 1;
    double S_target = n / (BIN_CAP / 2);
    uint32_t w = 63;
    while (w > 0 && ((double) c->twoM / (double) (1ull << w) < S_target
                || w > 32 + LOG_NB))
        w--;
    while (w < 63 && (c->twoM >> w) + 1 > MAX_SLICES) w++;
    c->w = w;
    const uint32_t oldS = c->S;
    c->S = (uint32_t) (((c->twoM - 1) >> w) + 1);
    if (c->S != oldS) c->cap_rq = 0;   /* d_gcnt, d_bins are cap_rq * S */
    if (c->S > MAX_SLICES || w > 32 + LOG_NB) return 0;

    CK(cudaSetDevice(c->device));
    cudaFree(c->d_pi); cudaFree(c->d_roots); cudaFree(c->d_pidx);
    CK(cudaMalloc(&c->d_pi, nprimes * sizeof(prime_info) + 1));
    CK(cudaMalloc(&c->d_roots, nroots * sizeof(uint64_t) + 1));
    CK(cudaMalloc(&c->d_pidx, nroots * sizeof(uint32_t) + 1));
    CK(cudaMemcpyAsync(c->d_pi, pi.data(), nprimes * sizeof(prime_info),
                cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyAsync(c->d_roots, roots, nroots * sizeof(uint64_t),
                cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyAsync(c->d_pidx, root_pidx, nroots * sizeof(uint32_t),
                cudaMemcpyHostToDevice, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    return 1;
}

static double now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

void psg_test(struct psg_ctx * c, const uint64_t * qs, uint32_t nq,
        const uint64_t * rq, const uint32_t * rq_qidx, uint32_t nrq,
        uint8_t * flags)
{
    const double t0 = now();
    CK(cudaSetDevice(c->device));
    memset(flags, 0, nrq);
    if (nrq == 0) return;

    /* batches: at most ~512 MB of bins, ~256 MB of inverses */
    const size_t per_rq = (size_t) c->S * BIN_CAP * sizeof(uint64_t);
    const size_t max_rq = std::max<size_t>(1, ((size_t) 512 << 20) / per_rq);
    const size_t max_q = std::max<size_t>(1,
            ((size_t) 256 << 20) / ((size_t) c->nprimes * sizeof(uint64_t) + 1));
    const uint32_t chunk = 4096;
    const uint32_t nchunks = (c->nroots + chunk - 1) / chunk;

    std::vector<uint32_t> rqq_rel;
    unsigned int overflow = 0;
    CK(cudaMemsetAsync(c->d_overflow, 0, sizeof(unsigned int), c->stream));

    for (uint32_t r0 = 0; r0 < nrq; ) {
        /* rq_qidx is non-decreasing: take rqs while the batch has room */
        const uint32_t q0 = rq_qidx[r0];
        uint32_t r1 = r0;
        while (r1 < nrq && r1 - r0 < max_rq && rq_qidx[r1] - q0 < max_q) r1++;
        const uint32_t q1 = rq_qidx[r1 - 1] + 1;
        const uint32_t nb = r1 - r0, nqb = q1 - q0;

        if (nqb > c->cap_q) {
            cudaFree(c->d_q); cudaFree(c->d_qbad); cudaFree(c->d_invR);
            c->cap_q = nqb;
            CK(cudaMalloc(&c->d_q, nqb * sizeof(uint64_t)));
            CK(cudaMalloc(&c->d_qbad, nqb));
            CK(cudaMalloc(&c->d_invR, (size_t) nqb * c->nprimes * sizeof(uint64_t) + 1));
        }
        rqq_rel.resize(nb);
        for (uint32_t j = 0; j < nb; j++) rqq_rel[j] = rq_qidx[r0 + j] - q0;

        CK(cudaMemcpyAsync(c->d_q, qs + q0, nqb * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, c->stream));
        CK(cudaMemsetAsync(c->d_qbad, 0, nqb, c->stream));
        {
            const uint64_t tot = (uint64_t) nqb * c->nprimes;
            inv_kernel<<<(unsigned) ((tot + THREADS - 1) / THREADS), THREADS, 0, c->stream>>>(
                    c->d_pi, c->nprimes, c->d_q, nqb, c->d_invR, c->d_qbad);
            CK(cudaGetLastError());
        }
        size_t cr = c->cap_rq;
        if (nb > cr) {
            cudaFree(c->d_rq); cudaFree(c->d_rqq); cudaFree(c->d_gcnt);
            cudaFree(c->d_bins); cudaFree(c->d_flags);
            cr = nb;
            CK(cudaMalloc(&c->d_rq, 2 * cr * sizeof(uint64_t)));
            CK(cudaMalloc(&c->d_rqq, cr * sizeof(uint32_t)));
            CK(cudaMalloc(&c->d_gcnt, cr * c->S * sizeof(uint32_t)));
            CK(cudaMalloc(&c->d_bins, cr * per_rq));
            CK(cudaMalloc(&c->d_flags, cr));
            c->cap_rq = cr;
        }
        CK(cudaMemcpyAsync(c->d_rq, rq + 2 * (size_t) r0, 2 * nb * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, c->stream));
        CK(cudaMemcpyAsync(c->d_rqq, rqq_rel.data(), nb * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, c->stream));
        CK(cudaMemsetAsync(c->d_gcnt, 0, (size_t) nb * c->S * sizeof(uint32_t), c->stream));
        CK(cudaMemsetAsync(c->d_flags, 0, nb, c->stream));

        stageA_kernel<<<dim3(nchunks, nb), THREADS, 2 * c->S * sizeof(uint32_t), c->stream>>>(
                c->d_pi, c->nprimes, c->d_roots, c->d_pidx, c->nroots, chunk,
                c->d_rq, c->d_rqq, c->d_invR, c->d_qbad,
                c->twoM, c->w, c->S, c->d_gcnt, c->d_bins);
        CK(cudaGetLastError());
        stageB_kernel<<<dim3(c->S, nb), THREADS, 0, c->stream>>>(
                c->d_rqq, c->d_qbad, c->S, c->d_gcnt, c->d_bins,
                c->d_flags, c->d_overflow);
        CK(cudaGetLastError());
        CK(cudaMemcpyAsync(flags + r0, c->d_flags, nb, cudaMemcpyDeviceToHost, c->stream));
        CK(cudaStreamSynchronize(c->stream));
        r0 = r1;
    }
    CK(cudaMemcpyAsync(&overflow, c->d_overflow, sizeof(unsigned int),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    unsigned long nf = 0;
    for (uint32_t j = 0; j < nrq; j++) nf += flags[j];
    c->rqs += nrq;
    c->flagged += nf;
    c->overflows += overflow;
    c->seconds += now() - t0;
}

void psg_ctx_stats(const struct psg_ctx * c, double * seconds,
        unsigned long * rqs, unsigned long * flagged, unsigned long * overflows)
{
    *seconds = c->seconds;
    *rqs = c->rqs;
    *flagged = c->flagged;
    *overflows = c->overflows;
}
