#ifndef CADO_MATMUL_CUDA_KERNELS_H
#define CADO_MATMUL_CUDA_KERNELS_H

/* GPU sparse matrix times vector over GF(2), for the bwc "cuda" matmul
 * backend (matmul-cuda.cpp). Plain C interface so that the C++ side
 * needs no CUDA headers.
 *
 * The matrix comes in the cado-nfs matrix_u32 layout: for each "major"
 * index i (a row, or a column if stored transposed), its length, then
 * that many "minor" indices. It is uploaded as is.
 *
 *   gather  (the fast direction): dst[i] = XOR of src[j] over row i
 *   scatter (the other one):      dst[j] ^= src[i] for every entry
 *
 * Vector elements are "words" 64-bit words wide (1 for b64, 2 for b128,
 * 4 for b256).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bwc_cuda_mat;

/* Picks the device itself (see the .cu file), uploads q. Returns NULL
 * and prints a message on failure. */
struct bwc_cuda_mat * bwc_cuda_mat_create(const uint32_t * q, size_t qsize,
        unsigned int nmajor, unsigned int nminor, unsigned int words);

void bwc_cuda_mat_free(struct bwc_cuda_mat * M);

/* Host vectors. gather != 0: src has nminor elements, dst nmajor.
 * gather == 0: src has nmajor elements, dst nminor. */
void bwc_cuda_mat_mul(struct bwc_cuda_mat * M, void * dst, const void * src,
        int gather);

/* Vectors kept on the GPU (krylov and mksol on a 1x1 grid, see
 * matmul_top_cuda_resident()). Everything runs on the matrix's device
 * and stream, in order; only bwc_cuda_download() and bwc_cuda_sync()
 * wait for it. */
void * bwc_cuda_alloc(struct bwc_cuda_mat * M, size_t bytes);
void bwc_cuda_free(struct bwc_cuda_mat * M, void * p);
void bwc_cuda_upload(struct bwc_cuda_mat * M, void * dev, const void * host,
        size_t bytes);
void bwc_cuda_download(struct bwc_cuda_mat * M, void * host, const void * dev,
        size_t bytes);
void bwc_cuda_zero(struct bwc_cuda_mat * M, void * dev, size_t bytes);
void bwc_cuda_sync(struct bwc_cuda_mat * M);
/* makes M's device the calling thread's current one (for NCCL setup) */
void bwc_cuda_set_device(struct bwc_cuda_mat * M);

/* device to device copy, queued */
void bwc_cuda_copy(struct bwc_cuda_mat * M, void * dst, const void * src,
        size_t bytes);

/* dst[i] = XOR over s < nslots of slots[s * nwords + i], i < nwords */
void bwc_cuda_xor_slots(struct bwc_cuda_mat * M, void * dst,
        const void * slots, unsigned int nslots, size_t nwords);

/* the matrix's stream (a cudaStream_t), for NCCL calls */
void * bwc_cuda_stream(struct bwc_cuda_mat * M);

/* bwc_cuda_mat_mul on device vectors */
void bwc_cuda_mul_dev(struct bwc_cuda_mat * M, void * dst, const void * src,
        int gather);

/* x_dotprod: dst[j] ^= XOR over t < nx of v[pos[j * nx + t]], j < m;
 * elements of the matrix's width; pos == UINT32_MAX entries are
 * skipped. dst, pos and v are device pointers. */
void bwc_cuda_xdot(struct bwc_cuda_mat * M, void * dst, const uint32_t * pos,
        unsigned int m, unsigned int nx, const void * v);

/* addmul_tiny: w[j] ^= u[j] * F for j < n, u of 64K bits, w of 64L
 * bits, F of 64K rows of L words (arith-cross.cpp). Device pointers. */
void bwc_cuda_addmul_tiny(struct bwc_cuda_mat * M, void * w, const void * u,
        const void * F, size_t n, unsigned int K, unsigned int L);

/* Device in use, and accumulated seconds in the kernel and in copies. */
int bwc_cuda_mat_device(const struct bwc_cuda_mat * M);
void bwc_cuda_mat_times(const struct bwc_cuda_mat * M, double * kernel,
        double * copies, unsigned long * calls);

#ifdef __cplusplus
}

/* Implemented by the cuda matmul backend, reached from matmul_top by
 * dynamic_cast. */
struct matmul_cuda_access {
    /* uploads the matrix if not done yet */
    virtual struct bwc_cuda_mat * cuda_mat() = 0;
    /* whether a product in direction d is the gather product */
    virtual int cuda_gather(int d) const = 0;
    virtual ~matmul_cuda_access() = default;
};
#endif

#endif
