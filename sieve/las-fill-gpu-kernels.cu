/* GPU top-level bucket fill, see las-fill-gpu-kernels.h.
 *
 * The arithmetic follows las-fbroot-qlattice.hpp (root transform),
 * las-plattice.hpp and las-reduce-plattice-simplistic.hpp (p-lattice
 * basis and its enumeration), so that the updates are exactly those of
 * fill_in_buckets_toplevel. The transformed root is the unique root of
 * the lattice modulo p, so computing it with plain 64-bit remainders
 * instead of REDC gives the same value.
 *
 * Kernels, for one side and one special-q:
 *   lfg_count   one thread per lattice: transform, reduce, walk, and
 *               count the hits per bucket in shared memory; each block
 *               covers lattices of one slice only
 *   lfg_sum     per (slice, bucket): total over the slice's blocks
 *   (scan)      exclusive sum over (slice, bucket), slice major
 *   lfg_offsets per (slice, bucket): where each block writes
 *   lfg_write   the walk again, writing the updates
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"
#include <cub/device/device_scan.cuh>

#include "las-fill-gpu-kernels.h"
#include "las-gpu-lattice.cuh"

#define CK(x) do {                                                      \
    cudaError_t const e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                            \
        fprintf(stderr, "las-fill-gpu: %s failed: %s\n", #x,            \
                cudaGetErrorString(e_));                                \
        abort();                                                        \
    }                                                                   \
} while (0)

#define LFG_BLOCK 128


struct lfg_block {
    uint32_t slice, first, count, pad;
};

struct lfg_table {
    lfg_lat * d_lats = nullptr;
    lfg_block * d_blocks = nullptr;
    size_t n = 0;
    uint32_t nslices = 0;
    uint32_t nblocks = 0;
    std::vector<uint32_t> slice_first_block;   /* nslices + 1 */
    uint32_t * d_slice_first_block = nullptr;
};

struct lfg_ctx {
    int device = 0;
    cudaStream_t stream;
    lfg_state * d_state = nullptr; size_t cap_state = 0;
    uint32_t * d_blockcounts = nullptr; size_t cap_bc = 0;
    uint32_t * d_counts = nullptr; size_t cap_counts = 0;
    uint32_t * d_base = nullptr; size_t cap_base = 0;
    uint8_t * d_scan_tmp = nullptr; size_t cap_scan = 0;
    uint8_t * d_updates = nullptr; size_t cap_upd = 0;
    lfg_special * d_spec = nullptr; uint32_t * d_nspec = nullptr; size_t cap_spec = 0;
    uint32_t * h_counts = nullptr; size_t cap_h_counts = 0;
    uint8_t * h_updates = nullptr; size_t cap_h_upd = 0;
    lfg_special * h_spec = nullptr; size_t cap_h_spec = 0;
    uint32_t * h_nspec = nullptr;
    double seconds = 0;
    unsigned long calls = 0;
};

/* {{{ kernels */
__global__ void lfg_count(const lfg_lat * lats, const lfg_block * blocks,
        lfg_qinfo Q, lfg_state * states, uint32_t * blockcounts,
        lfg_special * spec, uint32_t * nspec, uint32_t cap_spec)
{
    extern __shared__ uint32_t hist[];
    lfg_block const B = blocks[blockIdx.x];
    for (uint32_t b = threadIdx.x; b < Q.nbuckets; b += blockDim.x) hist[b] = 0;
    __syncthreads();

    uint32_t const t = threadIdx.x;
    if (t < B.count) {
        uint32_t const idx = B.first + t;
        lfg_state st;
        lfg_special sp;
        int const k = setup_lattice(lats[idx], Q, st, sp);
        if (k == 1) {
            uint32_t const maskI = (1u << Q.logI) - 1;
            uint64_t const even_mask = ((uint64_t) 1 << Q.logI) | 1;
            uint64_t const end = (uint64_t) Q.J << Q.logI;
            states[idx] = st;
            for (; st.x < end; step(st, maskI))
                if (st.x & even_mask)
                    atomicAdd(&hist[st.x >> Q.logB], 1u);
        } else {
            st.x = UINT64_MAX;
            states[idx] = st;
            if (k == 2) {
                uint32_t const s = atomicAdd(nspec, 1u);
                if (s < cap_spec) spec[s] = sp;
            }
        }
    }
    __syncthreads();
    uint32_t * out = blockcounts + (size_t) blockIdx.x * Q.nbuckets;
    for (uint32_t b = threadIdx.x; b < Q.nbuckets; b += blockDim.x) out[b] = hist[b];
}

/* counts[s][b] = sum of the slice's blocks */
__global__ void lfg_sum(const uint32_t * slice_first_block, uint32_t nslices,
        uint32_t nbuckets, const uint32_t * blockcounts, uint32_t * counts)
{
    uint32_t const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nslices * nbuckets) return;
    uint32_t const s = k / nbuckets, b = k % nbuckets;
    uint32_t sum = 0;
    for (uint32_t blk = slice_first_block[s]; blk < slice_first_block[s + 1]; blk++)
        sum += blockcounts[(size_t) blk * nbuckets + b];
    counts[k] = sum;
}

/* blockcounts[blk][b] <- where block blk writes its updates of bucket b */
__global__ void lfg_offsets(const uint32_t * slice_first_block, uint32_t nslices,
        uint32_t nbuckets, uint32_t * blockcounts, const uint32_t * base)
{
    uint32_t const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nslices * nbuckets) return;
    uint32_t const s = k / nbuckets, b = k % nbuckets;
    uint32_t off = base[k];
    for (uint32_t blk = slice_first_block[s]; blk < slice_first_block[s + 1]; blk++) {
        uint32_t const c = blockcounts[(size_t) blk * nbuckets + b];
        blockcounts[(size_t) blk * nbuckets + b] = off;
        off += c;
    }
}

template<int HINT_BYTES>
__global__ void lfg_write(const lfg_lat * lats, const lfg_block * blocks,
        lfg_qinfo Q, const lfg_state * states, const uint32_t * offsets,
        void * updates)
{
    extern __shared__ uint32_t cursor[];
    lfg_block const B = blocks[blockIdx.x];
    const uint32_t * in = offsets + (size_t) blockIdx.x * Q.nbuckets;
    for (uint32_t b = threadIdx.x; b < Q.nbuckets; b += blockDim.x) cursor[b] = in[b];
    __syncthreads();
    uint32_t const t = threadIdx.x;
    if (t >= B.count) return;
    uint32_t const idx = B.first + t;
    lfg_state st = states[idx];
    if (st.x == UINT64_MAX) return;
    uint32_t const hint = lats[idx].hint;
    uint32_t const maskI = (1u << Q.logI) - 1;
    uint64_t const even_mask = ((uint64_t) 1 << Q.logI) | 1;
    uint64_t const end = (uint64_t) Q.J << Q.logI;
    uint32_t const bmask = (1u << Q.logB) - 1;
    for (; st.x < end; step(st, maskI)) {
        if (!(st.x & even_mask)) continue;
        uint32_t const pos = atomicAdd(&cursor[st.x >> Q.logB], 1u);
        uint32_t const xx = (uint32_t) st.x & bmask;
        if (HINT_BYTES == 2)
            ((uint32_t *) updates)[pos] = hint | (xx << 16);
        else
            ((uint16_t *) updates)[pos] = (uint16_t) xx;
    }
}
/* }}} */

/* {{{ host */
template<typename T> static void grow_dev(T *& p, size_t & cap, size_t n)
{
    if (n <= cap) return;
    cudaFree(p);
    cap = n + n / 4 + 16;
    CK(cudaMalloc(&p, cap * sizeof(T)));
}
template<typename T> static void grow_host(T *& p, size_t & cap, size_t n)
{
    if (n <= cap) return;
    cudaFreeHost(p);
    cap = n + n / 4 + 16;
    CK(cudaMallocHost(&p, cap * sizeof(T)));
}

struct lfg_ctx * lfg_ctx_create(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    auto * c = new lfg_ctx;
    const char * e = getenv("CADO_LAS_GPU_DEVICE");
    if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!e) e = getenv("SLURM_LOCALID");
    c->device = (e ? atoi(e) : cado_gpu_claim("las", ndev)) % ndev;
    CK(cudaSetDevice(c->device));
    if (!cado_cuda_can_run("las GPU bucket fill")) { delete c; return nullptr; }
    CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    CK(cudaMallocHost(&c->h_nspec, sizeof(uint32_t)));
    CK(cudaMalloc(&c->d_nspec, sizeof(uint32_t)));
    return c;
}

void lfg_ctx_free(struct lfg_ctx * c)
{
    if (!c) return;
    cudaSetDevice(c->device);
    cudaFree(c->d_state); cudaFree(c->d_blockcounts); cudaFree(c->d_counts);
    cudaFree(c->d_base); cudaFree(c->d_scan_tmp); cudaFree(c->d_updates);
    cudaFree(c->d_spec); cudaFree(c->d_nspec);
    cudaFreeHost(c->h_counts); cudaFreeHost(c->h_updates); cudaFreeHost(c->h_spec);
    cudaFreeHost(c->h_nspec);
    cudaStreamDestroy(c->stream);
    delete c;
}

struct lfg_table * lfg_table_create(const struct lfg_lat * lats, size_t n,
        uint32_t nslices)
{
    auto * t = new lfg_table;
    t->n = n;
    t->nslices = nslices;
    /* blocks of at most LFG_BLOCK lattices of one slice */
    std::vector<lfg_block> blocks;
    t->slice_first_block.assign(nslices + 1, 0);
    size_t i = 0;
    for (uint32_t s = 0; s < nslices; s++) {
        t->slice_first_block[s] = blocks.size();
        while (i < n && lats[i].slice == s) {
            size_t j = i;
            while (j < n && lats[j].slice == s && j - i < LFG_BLOCK) j++;
            blocks.push_back({ s, (uint32_t) i, (uint32_t) (j - i), 0 });
            i = j;
        }
    }
    t->slice_first_block[nslices] = blocks.size();
    t->nblocks = blocks.size();
    if (n) {
        CK(cudaMalloc(&t->d_lats, n * sizeof(lfg_lat)));
        CK(cudaMemcpy(t->d_lats, lats, n * sizeof(lfg_lat), cudaMemcpyHostToDevice));
    }
    if (!blocks.empty()) {
        CK(cudaMalloc(&t->d_blocks, blocks.size() * sizeof(lfg_block)));
        CK(cudaMemcpy(t->d_blocks, blocks.data(), blocks.size() * sizeof(lfg_block),
                    cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&t->d_slice_first_block, (nslices + 1) * sizeof(uint32_t)));
    CK(cudaMemcpy(t->d_slice_first_block, t->slice_first_block.data(),
                (nslices + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return t;
}

void lfg_table_free(struct lfg_table * t)
{
    if (!t) return;
    cudaFree(t->d_lats);
    cudaFree(t->d_blocks);
    cudaFree(t->d_slice_first_block);
    delete t;
}

void lfg_fill(struct lfg_ctx * c, const struct lfg_table * t,
        const struct lfg_qinfo * Q,
        const uint32_t ** counts, const void ** updates, size_t * nupdates,
        const struct lfg_special ** specials, size_t * nspecials)
{
    cudaEvent_t e0, e1;
    CK(cudaSetDevice(c->device));
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, c->stream));

    uint32_t const nb = Q->nbuckets;
    size_t const ncnt = (size_t) t->nslices * nb;
    grow_dev(c->d_state, c->cap_state, t->n);
    grow_dev(c->d_blockcounts, c->cap_bc, (size_t) t->nblocks * nb);
    grow_dev(c->d_counts, c->cap_counts, ncnt);
    grow_dev(c->d_base, c->cap_base, ncnt);
    size_t const cap_spec = 4096;
    grow_dev(c->d_spec, c->cap_spec, cap_spec);
    grow_host(c->h_counts, c->cap_h_counts, ncnt);

    size_t const shmem = nb * sizeof(uint32_t);
    CK(cudaMemsetAsync(c->d_nspec, 0, sizeof(uint32_t), c->stream));
    if (t->nblocks)
        lfg_count<<<t->nblocks, LFG_BLOCK, shmem, c->stream>>>(t->d_lats,
                t->d_blocks, *Q, c->d_state, c->d_blockcounts, c->d_spec,
                c->d_nspec, (uint32_t) cap_spec);
    unsigned const g = (unsigned) ((ncnt + 255) / 256);
    if (ncnt)
        lfg_sum<<<g, 256, 0, c->stream>>>(t->d_slice_first_block, t->nslices,
                nb, c->d_blockcounts, c->d_counts);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(c->h_counts, c->d_counts, ncnt * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(c->h_nspec, c->d_nspec, sizeof(uint32_t),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));

    size_t total = 0;
    for (size_t k = 0; k < ncnt; k++) total += c->h_counts[k];
    size_t nsp = *c->h_nspec;
    if (nsp > cap_spec) {
        fprintf(stderr, "las-fill-gpu: %zu special lattices, more than %zu\n",
                nsp, cap_spec);
        abort();
    }

    int const ub = Q->hint_bytes ? 4 : 2;
    grow_dev(c->d_updates, c->cap_upd, total * ub);
    grow_host(c->h_updates, c->cap_h_upd, total * ub);
    grow_host(c->h_spec, c->cap_h_spec, nsp + 1);

    size_t tmp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp, c->d_counts, c->d_base, (int) ncnt, c->stream);
    grow_dev(c->d_scan_tmp, c->cap_scan, tmp);
    tmp = c->cap_scan;
    if (ncnt)
        CK(cub::DeviceScan::ExclusiveSum(c->d_scan_tmp, tmp, c->d_counts,
                    c->d_base, (int) ncnt, c->stream));
    if (ncnt)
        lfg_offsets<<<g, 256, 0, c->stream>>>(t->d_slice_first_block,
                t->nslices, nb, c->d_blockcounts, c->d_base);
    if (t->nblocks) {
        if (Q->hint_bytes)
            lfg_write<2><<<t->nblocks, LFG_BLOCK, shmem, c->stream>>>(t->d_lats,
                    t->d_blocks, *Q, c->d_state, c->d_blockcounts, c->d_updates);
        else
            lfg_write<0><<<t->nblocks, LFG_BLOCK, shmem, c->stream>>>(t->d_lats,
                    t->d_blocks, *Q, c->d_state, c->d_blockcounts, c->d_updates);
    }
    CK(cudaGetLastError());
    if (total)
        CK(cudaMemcpyAsync(c->h_updates, c->d_updates, total * ub,
                    cudaMemcpyDeviceToHost, c->stream));
    if (nsp)
        CK(cudaMemcpyAsync(c->h_spec, c->d_spec, nsp * sizeof(lfg_special),
                    cudaMemcpyDeviceToHost, c->stream));
    CK(cudaEventRecord(e1, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    c->seconds += ms * 1e-3;
    c->calls++;

    *counts = c->h_counts;
    *updates = c->h_updates;
    *nupdates = total;
    *specials = c->h_spec;
    *nspecials = nsp;
}

void lfg_ctx_stats(const struct lfg_ctx * c, double * seconds, unsigned long * calls)
{
    *seconds = c->seconds;
    *calls = c->calls;
}
/* }}} */

/* vim: set sw=4 fdm=marker: */
