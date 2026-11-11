/* GPU root sieve; see ropt_gpu_kernels.h */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"

#include "ropt_gpu_kernels.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {           \
    fprintf(stderr, "ropt GPU: %s failed at %s:%d: %s\n", #x,               \
            __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)

struct rsg_ctx {
    int device = 0;
    cudaStream_t stream = nullptr;
    rsg_op * h_ops = nullptr; size_t cap_ops = 0;
    rsg_op * d_ops = nullptr; size_t cap_dops = 0;
    int32_t * d_acc = nullptr; size_t cap_acc = 0;
    int16_t * d_out = nullptr;
    int16_t * h_out = nullptr; size_t cap_out = 0;
    double seconds = 0;
    unsigned long calls = 0, nops = 0;

    /* root counts */
    uint32_t * d_limbs = nullptr; size_t cap_limbs = 0;
    uint8_t * d_sign = nullptr; size_t cap_sign = 0;
    uint32_t * d_primes = nullptr; size_t cap_primes = 0;
    unsigned short * d_counts = nullptr; size_t cap_counts = 0;
};

/* one warp per line */
__global__ void lines_kernel(const rsg_op * ops, size_t n, int32_t * acc)
{
    const size_t w = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned lane = threadIdx.x & 31;
    if (w >= n) return;
    const rsg_op o = ops[w];
    const int64_t hits = (o.bound - o.start) / o.stride + 1;
    for (int64_t k = lane; k < hits; k += 32)
        atomicAdd(&acc[o.start + k * o.stride], o.sub);
}

__global__ void finish_kernel(const int32_t * acc, int16_t * out, size_t len,
        int32_t init)
{
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < len;
            i += (size_t) gridDim.x * blockDim.x)
        out[i] = (int16_t) (init - acc[i]);
}

/* block = (prime k, polynomial): coefficients mod p, then f(x) for every
   x mod p */
__global__ void rootcount_kernel(int deg, int nlimbs, const uint32_t * limbs,
        const uint8_t * sign, int nprimes, const uint32_t * primes,
        unsigned short * counts)
{
    __shared__ uint32_t c[32];
    __shared__ unsigned int n;
    const int k = blockIdx.x, poly = blockIdx.y;
    const uint32_t p = primes[k];
    if (threadIdx.x == 0) n = 0;
    if ((int) threadIdx.x <= deg) {
        const int i = threadIdx.x;
        const uint32_t * L = limbs + ((size_t) poly * (deg + 1) + i) * nlimbs;
        uint64_t r = 0;
        for (int l = nlimbs; l--; )
            r = ((r << 32) | L[l]) % p;
        if (sign[(size_t) poly * (deg + 1) + i] && r)
            r = p - r;
        c[i] = (uint32_t) r;
    }
    __syncthreads();
    unsigned int mine = 0;
    for (uint32_t x = threadIdx.x; x < p; x += blockDim.x) {
        uint32_t v = c[deg];
        for (int i = deg; i--; )
            v = (v * x + c[i]) % p;
        mine += v == 0;
    }
    if (mine) atomicAdd(&n, mine);
    __syncthreads();
    if (threadIdx.x == 0)
        counts[(size_t) poly * nprimes + k] = (unsigned short) n;
}

template<typename T> static void ensure(T *& p, size_t & cap, size_t n)
{
    if (n <= cap) return;
    cudaFree(p);
    cap = n + n / 2 + 64;
    CK(cudaMalloc(&p, cap * sizeof(T)));
}

static double now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

struct rsg_ctx * rsg_ctx_create(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    auto * c = new rsg_ctx;
    const char * e = getenv("CADO_POLYSELECT_GPU_DEVICE");
    if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!e) e = getenv("SLURM_LOCALID");
    c->device = (e ? atoi(e) : cado_gpu_claim("polyselect", ndev)) % ndev;
    CK(cudaSetDevice(c->device));
    if (!cado_cuda_can_run("polyselect GPU root optimization")) { delete c; return nullptr; }
    CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    return c;
}

void rsg_ctx_free(struct rsg_ctx * c)
{
    if (!c) return;
    cudaSetDevice(c->device);
    cudaFreeHost(c->h_ops); cudaFree(c->d_ops); cudaFree(c->d_acc);
    cudaFree(c->d_out); cudaFreeHost(c->h_out);
    cudaFree(c->d_limbs); cudaFree(c->d_sign); cudaFree(c->d_primes);
    cudaFree(c->d_counts);
    cudaStreamDestroy(c->stream);
    delete c;
}

struct rsg_op * rsg_ops_buffer(struct rsg_ctx * c, size_t n)
{
    if (n > c->cap_ops) {
        CK(cudaSetDevice(c->device));
        rsg_op * p;
        size_t cap = n + n / 2 + 1024;
        CK(cudaMallocHost(&p, cap * sizeof(rsg_op)));
        if (c->h_ops) {
            memcpy(p, c->h_ops, c->cap_ops * sizeof(rsg_op));
            cudaFreeHost(c->h_ops);
        }
        c->h_ops = p;
        c->cap_ops = cap;
    }
    return c->h_ops;
}

void rsg_apply(struct rsg_ctx * c, size_t n, int16_t init, int16_t * out,
        size_t len)
{
    const double t0 = now();
    CK(cudaSetDevice(c->device));
    if (n > c->cap_dops) {
        cudaFree(c->d_ops);
        c->cap_dops = n + n / 2 + 1024;
        CK(cudaMalloc(&c->d_ops, c->cap_dops * sizeof(rsg_op)));
    }
    if (len > c->cap_acc) {
        cudaFree(c->d_acc); cudaFree(c->d_out); cudaFreeHost(c->h_out);
        c->cap_acc = len;
        CK(cudaMalloc(&c->d_acc, len * sizeof(int32_t)));
        CK(cudaMalloc(&c->d_out, len * sizeof(int16_t)));
        CK(cudaMallocHost(&c->h_out, len * sizeof(int16_t)));
    }
    CK(cudaMemsetAsync(c->d_acc, 0, len * sizeof(int32_t), c->stream));
    if (n) {
        CK(cudaMemcpyAsync(c->d_ops, c->h_ops, n * sizeof(rsg_op),
                    cudaMemcpyHostToDevice, c->stream));
        const unsigned threads = 256;
        const size_t blocks = (n * 32 + threads - 1) / threads;
        lines_kernel<<<(unsigned) blocks, threads, 0, c->stream>>>(c->d_ops, n, c->d_acc);
        CK(cudaGetLastError());
    }
    finish_kernel<<<1024, 256, 0, c->stream>>>(c->d_acc, c->d_out, len, init);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(c->h_out, c->d_out, len * sizeof(int16_t),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    memcpy(out, c->h_out, len * sizeof(int16_t));
    c->calls++;
    c->nops += n;
    c->seconds += now() - t0;
}

void rsg_rootcounts(struct rsg_ctx * c, int npolys, int deg, int nlimbs,
        const uint32_t * limbs, const uint8_t * sign,
        int nprimes, const uint32_t * primes, unsigned short * counts)
{
    CK(cudaSetDevice(c->device));
    const size_t ncoef = (size_t) npolys * (deg + 1);
    ensure(c->d_limbs, c->cap_limbs, ncoef * nlimbs);
    ensure(c->d_sign, c->cap_sign, ncoef);
    ensure(c->d_primes, c->cap_primes, (size_t) nprimes);
    ensure(c->d_counts, c->cap_counts, (size_t) npolys * nprimes);
    CK(cudaMemcpyAsync(c->d_limbs, limbs, ncoef * nlimbs * sizeof(uint32_t),
                cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyAsync(c->d_sign, sign, ncoef, cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyAsync(c->d_primes, primes, nprimes * sizeof(uint32_t),
                cudaMemcpyHostToDevice, c->stream));
    rootcount_kernel<<<dim3(nprimes, npolys), 128, 0, c->stream>>>(deg, nlimbs,
            c->d_limbs, c->d_sign, nprimes, c->d_primes, c->d_counts);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(counts, c->d_counts, (size_t) npolys * nprimes * sizeof(unsigned short),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
}

void rsg_ctx_stats(const struct rsg_ctx * c, double * seconds,
        unsigned long * calls, unsigned long * ops)
{
    *seconds = c->seconds;
    *calls = c->calls;
    *ops = c->nops;
}
