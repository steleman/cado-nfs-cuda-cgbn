/* GPU sparse matrix times vector over GF(2) for bwc (see
 * matmul-cuda-kernels.h).
 *
 * Work is cut into items of at most CHUNK indices: a row up to CHUNK
 * long is one item, a longer one several (row lengths are very uneven:
 * on a c120 block the median is 33, the longest 182,916, and one warp on
 * that row used to finish long after the rest). The long rows' items
 * come first so they start early.
 *
 * Gather: one warp per item. The lanes read the indices with coalesced
 * streaming loads, XOR the source elements they name, and the warp
 * combines the 32 partial sums with shuffles; a row's only item stores
 * its result, the items of a long row XOR theirs in atomically. The cost
 * is the index stream (0.40 ms of the 0.95 ms per product on the c120
 * block) plus the random reads from the source vector, which stays in
 * L2. A read fetches a 32-byte sector, so wider elements cost little
 * more: b128 the same as b64, b256 1.3 times.
 *
 * Scatter: same walk, but each lane XORs src[i] into dst[j] atomically.
 * bwc only uses it outside the iterations (the matrix is stored for the
 * gather direction).
 *
 * Device choice: the n-th matrix created in a process goes to device
 * (local_rank + n * local_size) % ndev, where local_rank/local_size come
 * from the MPI launcher's environment (Open MPI, MPICH, Slurm) and are
 * 0/1 without one. So T threads in one process spread over the GPUs, and
 * so do R ranks on a node with one thread each. CADO_BWC_CUDA_DEVICE
 * overrides the choice (a fixed device for every matrix).
 */

#include "matmul-cuda-kernels.h"

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do {                                              \
        cudaError_t const e_ = (x);                                     \
        if (e_ != cudaSuccess) {                                        \
            fprintf(stderr, "bwc cuda: %s failed at %s:%d: %s\n",       \
                    #x, __FILE__, __LINE__, cudaGetErrorString(e_));    \
            abort();                                                    \
        }                                                               \
    } while (0)

/* at most this many indices per work item */
#define CHUNK 4096

struct item {
    uint64_t begin;         /* offset of the first index in q */
    uint32_t len;           /* <= CHUNK */
    uint32_t row;           /* major index; top bit: XOR atomically */
};
#define ITEM_ATOMIC 0x80000000u

struct bwc_cuda_mat {
    int dev;
    unsigned int nmajor, nminor, words;
    uint32_t * q;           /* device copy of the matrix_u32 data */
    item * items;
    unsigned int nitems;
    int has_split;          /* some row has several items */
    uint64_t * vin;         /* device vectors, max(nmajor,nminor) elts */
    uint64_t * vout;
    cudaStream_t stream;
    cudaEvent_t ev[4];
    int pending;            /* a device product's events not read yet */
    double t_kernel, t_copies;
    unsigned long calls;
};

static int env_int(const char * const * names, int dflt)
{
    for ( ; *names ; names++) {
        const char * v = getenv(*names);
        if (v && *v) return atoi(v);
    }
    return dflt;
}

static int pick_device()
{
    static std::atomic<int> counter { 0 };
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    const char * fixed = getenv("CADO_BWC_CUDA_DEVICE");
    if (fixed && *fixed) return atoi(fixed) % ndev;
    static const char * const lr[] = { "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID", "SLURM_LOCALID", nullptr };
    static const char * const ls[] = { "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS", "SLURM_NTASKS_PER_NODE", nullptr };
    int const local_rank = env_int(lr, 0);
    int const local_size = env_int(ls, 1);
    int const n = counter++;
    return (local_rank + n * local_size) % ndev;
}

/* XOR-reduce a 64-bit value across the warp */
static __device__ __forceinline__ uint64_t warp_xor(uint64_t v)
{
    for (int o = 16 ; o ; o >>= 1)
        v ^= __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template<int W> struct vload;
template<> struct vload<1> {
    static __device__ __forceinline__ void x(uint64_t * a, const uint64_t * p)
    { a[0] ^= __ldg(p); }
};
template<> struct vload<2> {
    static __device__ __forceinline__ void x(uint64_t * a, const uint64_t * p)
    {
        ulonglong2 const v = __ldg((const ulonglong2 *) p);
        a[0] ^= v.x; a[1] ^= v.y;
    }
};
template<> struct vload<4> {
    static __device__ __forceinline__ void x(uint64_t * a, const uint64_t * p)
    {
        ulonglong2 const v = __ldg((const ulonglong2 *) p);
        ulonglong2 const u = __ldg((const ulonglong2 *) p + 1);
        a[0] ^= v.x; a[1] ^= v.y; a[2] ^= u.x; a[3] ^= u.y;
    }
};

template<int W>
static __global__ void __launch_bounds__(256)
gather_kernel(const uint32_t * __restrict__ q,
        const item * __restrict__ items, unsigned int nitems,
        const uint64_t * __restrict__ src, uint64_t * __restrict__ dst)
{
    unsigned int const w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int const lane = threadIdx.x & 31;
    if (w >= nitems) return;
    item const it = items[w];
    const uint32_t * qq = q + it.begin;
    uint64_t acc[W];
#pragma unroll
    for (int u = 0 ; u < W ; u++) acc[u] = 0;
    unsigned int k = lane;
    /* two loads in flight per lane */
    for ( ; k + 32 < it.len ; k += 64) {
        uint32_t const j0 = __ldcs(qq + k);
        uint32_t const j1 = __ldcs(qq + k + 32);
        vload<W>::x(acc, src + (size_t) j0 * W);
        vload<W>::x(acc, src + (size_t) j1 * W);
    }
    if (k < it.len)
        vload<W>::x(acc, src + (size_t) __ldcs(qq + k) * W);
#pragma unroll
    for (int u = 0 ; u < W ; u++) acc[u] = warp_xor(acc[u]);
    if (lane == 0) {
        size_t const r = (size_t) (it.row & ~ITEM_ATOMIC) * W;
        if (it.row & ITEM_ATOMIC) {
#pragma unroll
            for (int u = 0 ; u < W ; u++)
                atomicXor((unsigned long long *) dst + r + u,
                        (unsigned long long) acc[u]);
        } else {
#pragma unroll
            for (int u = 0 ; u < W ; u++) dst[r + u] = acc[u];
        }
    }
}

/* any width: one word per pass */
static __global__ void __launch_bounds__(256)
gather_kernel_any(const uint32_t * __restrict__ q,
        const item * __restrict__ items, unsigned int nitems,
        unsigned int words,
        const uint64_t * __restrict__ src, uint64_t * __restrict__ dst)
{
    unsigned int const w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int const lane = threadIdx.x & 31;
    if (w >= nitems) return;
    item const it = items[w];
    const uint32_t * qq = q + it.begin;
    size_t const r = (size_t) (it.row & ~ITEM_ATOMIC) * words;
    for (unsigned int u = 0 ; u < words ; u++) {
        uint64_t acc = 0;
        for (unsigned int k = lane ; k < it.len ; k += 32)
            acc ^= src[(size_t) qq[k] * words + u];
        acc = warp_xor(acc);
        if (lane == 0) {
            if (it.row & ITEM_ATOMIC)
                atomicXor((unsigned long long *) dst + r + u,
                        (unsigned long long) acc);
            else
                dst[r + u] = acc;
        }
    }
}

static __global__ void __launch_bounds__(256)
scatter_kernel(const uint32_t * __restrict__ q,
        const item * __restrict__ items, unsigned int nitems,
        unsigned int words,
        const uint64_t * __restrict__ src, uint64_t * __restrict__ dst)
{
    unsigned int const w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int const lane = threadIdx.x & 31;
    if (w >= nitems) return;
    item const it = items[w];
    const uint32_t * qq = q + it.begin;
    size_t const r = (size_t) (it.row & ~ITEM_ATOMIC) * words;
    for (unsigned int u = 0 ; u < words ; u++) {
        uint64_t const v = src[r + u];
        if (!v) continue;
        for (unsigned int k = lane ; k < it.len ; k += 32)
            atomicXor((unsigned long long *) dst + (size_t) qq[k] * words + u,
                    (unsigned long long) v);
    }
}

extern "C" struct bwc_cuda_mat * bwc_cuda_mat_create(const uint32_t * q,
        size_t qsize, unsigned int nmajor, unsigned int nminor,
        unsigned int words)
{
    /* work items, the pieces of split rows first */
    std::vector<item> items, light;
    light.reserve(nmajor);
    size_t pos = 0;
    for (unsigned int i = 0 ; i < nmajor ; i++) {
        if (pos >= qsize || pos + 1 + (size_t) q[pos] > qsize) {
            fprintf(stderr, "bwc cuda: matrix data too short\n");
            return nullptr;
        }
        size_t const len = q[pos];
        size_t const b = pos + 1;
        if (len <= CHUNK) {
            light.push_back(item { b, (uint32_t) len, i });
        } else {
            for (size_t c = 0 ; c < len ; c += CHUNK)
                items.push_back(item { b + c,
                        (uint32_t) (len - c < CHUNK ? len - c : CHUNK),
                        i | ITEM_ATOMIC });
        }
        pos = b + len;
    }
    if (pos != qsize) {
        fprintf(stderr, "bwc cuda: matrix data has %zu words, expected %zu\n",
                qsize, pos);
        return nullptr;
    }
    if (nmajor & ITEM_ATOMIC) {
        fprintf(stderr, "bwc cuda: too many rows (%u)\n", nmajor);
        return nullptr;
    }
    int const has_split = !items.empty();
    items.insert(items.end(), light.begin(), light.end());
    std::vector<item>().swap(light);

    auto * M = new bwc_cuda_mat();
    M->dev = pick_device();
    M->nmajor = nmajor;
    M->nminor = nminor;
    M->words = words;
    CUDA_CHECK(cudaSetDevice(M->dev));
    /* bwc chose mm_impl=cuda: there is no CPU fallback from here */
    if (!cado_cuda_can_run("bwc cuda matmul")) {
        fprintf(stderr, "# bwc: mm_impl=cuda cannot run on this GPU; rebuild"
                " for it, or use another mm_impl\n");
        exit(EXIT_FAILURE);
    }
    size_t const vmax = (size_t) (nmajor > nminor ? nmajor : nminor) * words;
    CUDA_CHECK(cudaMalloc(&M->q, (qsize ? qsize : 1) * sizeof(uint32_t)));
    M->nitems = items.size();
    M->has_split = has_split;
    CUDA_CHECK(cudaMalloc(&M->items, (items.empty() ? 1 : items.size()) * sizeof(item)));
    CUDA_CHECK(cudaMalloc(&M->vin, (vmax ? vmax : 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&M->vout, (vmax ? vmax : 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(M->q, q, qsize * sizeof(uint32_t),
                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M->items, items.data(),
                items.size() * sizeof(item), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&M->stream, cudaStreamNonBlocking));
    for (auto & e : M->ev) CUDA_CHECK(cudaEventCreate(&e));
    return M;
}

extern "C" void bwc_cuda_mat_free(struct bwc_cuda_mat * M)
{
    if (!M) return;
    cudaSetDevice(M->dev);
    cudaStreamSynchronize(M->stream);
    for (auto & e : M->ev) cudaEventDestroy(e);
    cudaStreamDestroy(M->stream);
    cudaFree(M->q);
    cudaFree(M->items);
    cudaFree(M->vin);
    cudaFree(M->vout);
    delete M;
}

/* the product on the device buffers, queued on M's stream */
static void launch_mul(struct bwc_cuda_mat * M, uint64_t * dst,
        const uint64_t * src, int gather)
{
    size_t const W = M->words;
    size_t const nout = (gather ? M->nmajor : M->nminor) * W * sizeof(uint64_t);
    unsigned int const threads = 256;
    unsigned int const blocks = (unsigned int)
        (((size_t) M->nitems * 32 + threads - 1) / threads);
    cudaStream_t const s = M->stream;
    /* split rows and the scatter accumulate; rows without entries are
     * items too, so the plain gather writes every element */
    if (!gather || M->has_split || !M->nitems)
        CUDA_CHECK(cudaMemsetAsync(dst, 0, nout, s));
    if (!blocks) return;
    if (gather) {
        switch (W) {
        case 1: gather_kernel<1><<<blocks, threads, 0, s>>>(M->q,
                        M->items, M->nitems, src, dst); break;
        case 2: gather_kernel<2><<<blocks, threads, 0, s>>>(M->q,
                        M->items, M->nitems, src, dst); break;
        case 4: gather_kernel<4><<<blocks, threads, 0, s>>>(M->q,
                        M->items, M->nitems, src, dst); break;
        default: gather_kernel_any<<<blocks, threads, 0, s>>>(M->q,
                        M->items, M->nitems, M->words, src, dst);
        }
    } else {
        scatter_kernel<<<blocks, threads, 0, s>>>(M->q, M->items,
                M->nitems, M->words, src, dst);
    }
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void bwc_cuda_mat_mul(struct bwc_cuda_mat * M, void * dst,
        const void * src, int gather)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    bwc_cuda_sync(M);
    size_t const W = M->words;
    size_t const nin = (gather ? M->nminor : M->nmajor) * W * sizeof(uint64_t);
    size_t const nout = (gather ? M->nmajor : M->nminor) * W * sizeof(uint64_t);
    cudaStream_t const s = M->stream;

    CUDA_CHECK(cudaEventRecord(M->ev[0], s));
    CUDA_CHECK(cudaMemcpyAsync(M->vin, src, nin, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaEventRecord(M->ev[1], s));
    launch_mul(M, M->vout, M->vin, gather);
    CUDA_CHECK(cudaEventRecord(M->ev[2], s));
    CUDA_CHECK(cudaMemcpyAsync(dst, M->vout, nout, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaEventRecord(M->ev[3], s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    float a, b, c;
    cudaEventElapsedTime(&a, M->ev[0], M->ev[1]);
    cudaEventElapsedTime(&b, M->ev[1], M->ev[2]);
    cudaEventElapsedTime(&c, M->ev[2], M->ev[3]);
    M->t_copies += (a + c) * 1e-3;
    M->t_kernel += b * 1e-3;
    M->calls++;
}

/* {{{ vectors on the device */
extern "C" void * bwc_cuda_alloc(struct bwc_cuda_mat * M, size_t bytes)
{
    void * p = nullptr;
    CUDA_CHECK(cudaSetDevice(M->dev));
    CUDA_CHECK(cudaMalloc(&p, bytes ? bytes : 1));
    return p;
}

extern "C" void bwc_cuda_free(struct bwc_cuda_mat * M, void * p)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    bwc_cuda_sync(M);
    cudaFree(p);
}

extern "C" void bwc_cuda_upload(struct bwc_cuda_mat * M, void * dev,
        const void * host, size_t bytes)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    /* pageable source: returns once the data is staged */
    CUDA_CHECK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice,
                M->stream));
}

extern "C" void bwc_cuda_download(struct bwc_cuda_mat * M, void * host,
        const void * dev, size_t bytes)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    CUDA_CHECK(cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost,
                M->stream));
    bwc_cuda_sync(M);
}

extern "C" void bwc_cuda_zero(struct bwc_cuda_mat * M, void * dev,
        size_t bytes)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    CUDA_CHECK(cudaMemsetAsync(dev, 0, bytes, M->stream));
}

extern "C" void bwc_cuda_sync(struct bwc_cuda_mat * M)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    CUDA_CHECK(cudaStreamSynchronize(M->stream));
    if (M->pending) {
        float b;
        CUDA_CHECK(cudaEventElapsedTime(&b, M->ev[1], M->ev[2]));
        M->t_kernel += b * 1e-3;
        M->calls++;
        M->pending = 0;
    }
}

extern "C" void bwc_cuda_mul_dev(struct bwc_cuda_mat * M, void * dst,
        const void * src, int gather)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    /* one product timed at a time */
    if (M->pending) bwc_cuda_sync(M);
    CUDA_CHECK(cudaEventRecord(M->ev[1], M->stream));
    launch_mul(M, (uint64_t *) dst, (const uint64_t *) src, gather);
    CUDA_CHECK(cudaEventRecord(M->ev[2], M->stream));
    M->pending = 1;
}

extern "C" void bwc_cuda_copy(struct bwc_cuda_mat * M, void * dst,
        const void * src, size_t bytes)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    if (bytes)
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                    M->stream));
}

static __global__ void xor_slots_kernel(uint64_t * __restrict__ dst,
        const uint64_t * __restrict__ slots, unsigned int nslots,
        size_t nwords)
{
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x ;
            i < nwords ; i += (size_t) gridDim.x * blockDim.x) {
        uint64_t a = 0;
        for (unsigned int k = 0 ; k < nslots ; k++)
            a ^= slots[(size_t) k * nwords + i];
        dst[i] = a;
    }
}

extern "C" void bwc_cuda_xor_slots(struct bwc_cuda_mat * M, void * dst,
        const void * slots, unsigned int nslots, size_t nwords)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    if (!nwords) return;
    size_t const want = (nwords + 255) / 256;
    unsigned int const blocks = (unsigned int) (want < 4096 ? want : 4096);
    xor_slots_kernel<<<blocks, 256, 0, M->stream>>>((uint64_t *) dst,
            (const uint64_t *) slots, nslots, nwords);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void bwc_cuda_set_device(struct bwc_cuda_mat * M)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
}

extern "C" void * bwc_cuda_stream(struct bwc_cuda_mat * M)
{
    return (void *) M->stream;
}

static __global__ void xdot_kernel(uint64_t * __restrict__ dst,
        const uint32_t * __restrict__ pos, unsigned int m, unsigned int nx,
        unsigned int words, const uint64_t * __restrict__ v)
{
    unsigned int const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m * words) return;
    unsigned int const j = t / words, u = t % words;
    uint64_t acc = 0;
    for (unsigned int k = 0 ; k < nx ; k++) {
        uint32_t const p = pos[j * nx + k];
        if (p != UINT32_MAX) acc ^= v[(size_t) p * words + u];
    }
    dst[t] ^= acc;
}

extern "C" void bwc_cuda_xdot(struct bwc_cuda_mat * M, void * dst,
        const uint32_t * pos, unsigned int m, unsigned int nx, const void * v)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    unsigned int const n = m * M->words;
    if (!n) return;
    xdot_kernel<<<(n + 255) / 256, 256, 0, M->stream>>>((uint64_t *) dst,
            pos, m, nx, M->words, (const uint64_t *) v);
    CUDA_CHECK(cudaGetLastError());
}

/* addmul_tiny with 4-bit tables in shared memory: table t (16K of them)
 * holds the 16 combinations of F's rows 4t..4t+3, L words each, so
 * that is 16K * 16 * L words (32 KB for K = L = 4). */
template<int K, int L>
static __global__ void __launch_bounds__(256)
addmul_tiny_kernel(uint64_t * __restrict__ w, const uint64_t * __restrict__ u,
        const uint64_t * __restrict__ F, size_t n)
{
    extern __shared__ uint64_t tab[];
    for (unsigned int e = threadIdx.x ; e < 16 * K * 16 * L ; e += blockDim.x) {
        unsigned int const l = e % L;
        unsigned int const x = (e / L) % 16;
        unsigned int const t = e / (16 * L);
        uint64_t a = 0;
        for (unsigned int b = 0 ; b < 4 ; b++)
            if (x >> b & 1) a ^= F[(size_t) (4 * t + b) * L + l];
        tab[e] = a;
    }
    __syncthreads();
    for (size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x ; j < n ;
            j += (size_t) gridDim.x * blockDim.x) {
        uint64_t acc[L];
#pragma unroll
        for (int l = 0 ; l < L ; l++) acc[l] = 0;
#pragma unroll
        for (int k = 0 ; k < K ; k++) {
            uint64_t a = u[j * K + k];
#pragma unroll
            for (int nib = 0 ; nib < 16 ; nib++, a >>= 4) {
                const uint64_t * e = tab + ((k * 16 + nib) * 16 + (a & 15)) * L;
#pragma unroll
                for (int l = 0 ; l < L ; l++) acc[l] ^= e[l];
            }
        }
#pragma unroll
        for (int l = 0 ; l < L ; l++) w[j * L + l] ^= acc[l];
    }
}

/* any K, L: bit by bit */
static __global__ void __launch_bounds__(256)
addmul_tiny_any(uint64_t * __restrict__ w, const uint64_t * __restrict__ u,
        const uint64_t * __restrict__ F, size_t n, unsigned int K,
        unsigned int L)
{
    for (size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x ; j < n ;
            j += (size_t) gridDim.x * blockDim.x) {
        for (unsigned int l = 0 ; l < L ; l++) {
            uint64_t acc = 0;
            for (unsigned int k = 0 ; k < K ; k++) {
                uint64_t const a = u[j * K + k];
                for (unsigned int b = 0 ; b < 64 ; b++)
                    if (a >> b & 1) acc ^= F[(size_t) (64 * k + b) * L + l];
            }
            w[j * L + l] ^= acc;
        }
    }
}

extern "C" void bwc_cuda_addmul_tiny(struct bwc_cuda_mat * M, void * w,
        const void * u, const void * F, size_t n, unsigned int K,
        unsigned int L)
{
    CUDA_CHECK(cudaSetDevice(M->dev));
    if (!n) return;
    int dev_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, M->dev));
    /* enough blocks to fill the GPU, each building its tables once */
    size_t const want = (n + 255) / 256;
    unsigned int const blocks = (unsigned int)
        (want < (size_t) dev_sms * 8 ? want : (size_t) dev_sms * 8);
    size_t const shm = (size_t) 16 * K * 16 * L * sizeof(uint64_t);
    auto * xw = (uint64_t *) w;
    auto const * xu = (const uint64_t *) u;
    auto const * xF = (const uint64_t *) F;
    cudaStream_t const s = M->stream;
#define AMT(k_, l_) else if (K == k_ && L == l_)                            \
        addmul_tiny_kernel<k_, l_><<<blocks, 256, shm, s>>>(xw, xu, xF, n)
    if (0) {}
    AMT(1,1); AMT(1,2); AMT(2,1); AMT(2,2);
    AMT(1,4); AMT(4,1); AMT(2,4); AMT(4,2); AMT(4,4);
    else addmul_tiny_any<<<blocks, 256, 0, s>>>(xw, xu, xF, n, K, L);
#undef AMT
    CUDA_CHECK(cudaGetLastError());
}
/* }}} */

extern "C" int bwc_cuda_mat_device(const struct bwc_cuda_mat * M)
{
    return M->dev;
}

extern "C" void bwc_cuda_mat_times(const struct bwc_cuda_mat * M,
        double * kernel, double * copies, unsigned long * calls)
{
    *kernel = M->t_kernel;
    *copies = M->t_copies;
    *calls = M->calls;
}
