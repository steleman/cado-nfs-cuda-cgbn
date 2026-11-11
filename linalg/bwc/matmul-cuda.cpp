#include "cado.h" // IWYU pragma: keep

/* bwc matmul backend on an NVIDIA GPU (GF(2) only).
 *
 * The local matrix block is kept on the GPU in the matrix_u32 layout
 * (see matmul-cuda-kernels.cu); mul() copies the source vector in, runs
 * the product and copies the result back. Select it with mm_impl=cuda.
 * One GPU per matmul instance: with thr=1 the GPU does the whole matrix,
 * with a thread grid or several MPI ranks the blocks are spread over the
 * node's GPUs.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <utility>
#include <vector>

#include "arith-generic.hpp"
#include "arith-hard.hpp"
#include "macros.h"
#include "matmul.hpp"
#include "matmul-common.hpp"
#include "matmul-cuda-kernels.h"
#include "matrix_u32.hpp"
#include "params.hpp"

#define MM_EXTENSION   "-cuda"
#define MM_MAGIC_FAMILY        0xa00cUL
#define MM_MAGIC_VERSION       0x1001UL
#define MM_MAGIC (MM_MAGIC_FAMILY << 16 | MM_MAGIC_VERSION)

/* Same convention as matmul-basic: the stored layout is chosen so that
 * the optimized direction is the gather (row by row) product. */
#define MM_DIR0_PREFERS_TRANSP_MULT   1

template<typename Arith>
struct matmul_cuda : public matmul_interface, public matmul_cuda_access {
    Arith * xab;
    std::vector<uint32_t> q;
    bwc_cuda_mat * M = nullptr;

    void build_cache(matrix_u32 &&) override;
    int reload_cache_private() override;
    void save_cache_private() override;
    void mul(void *, const void *, int) override;
    void report(double) override;

    bwc_cuda_mat * cuda_mat() override { if (!M) upload(); return M; }
    int cuda_gather(int d) const override { return d == !store_transposed; }

    matmul_cuda(matmul_public &&, arith_concrete_base *, cxx_param_list &, int);
    ~matmul_cuda() override;

    matmul_cuda(matmul_cuda const &) = delete;
    matmul_cuda& operator=(matmul_cuda const &) = delete;
    matmul_cuda(matmul_cuda &&) noexcept = delete;
    matmul_cuda& operator=(matmul_cuda &&) noexcept = delete;

  private:
    void upload();
};

template<typename Arith>
matmul_cuda<Arith>::matmul_cuda(matmul_public && P, arith_concrete_base * pxx, cxx_param_list & pl, int optimized_direction)
    : matmul_interface(std::move(P))
    , xab((Arith *) pxx) // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
{
    int const suggest = optimized_direction ^ MM_DIR0_PREFERS_TRANSP_MULT;
    store_transposed = suggest;
    pl.parse("mm_store_transposed", store_transposed);
    if (store_transposed != suggest) {
        fprintf(stderr, "Warning, mm_store_transposed"
                " overrides suggested matrix storage ordering\n");
    }
    if (xab->elt_stride() % sizeof(uint64_t)) {
        fprintf(stderr, "mm_impl=cuda: element size %zu is not a multiple"
                " of 64 bits (GF(2) only)\n", (size_t) xab->elt_stride());
        abort();
    }
}

template<typename Arith>
matmul_cuda<Arith>::~matmul_cuda()
{
    if (M) {
        double k, c;
        unsigned long n;
        bwc_cuda_mat_times(M, &k, &c, &n);
        if (n && getenv("CADO_BWC_CUDA_VERBOSE"))
            fprintf(stderr, "mm_impl=cuda: GPU %d, %lu products,"
                    " kernel %.3f s, copies %.3f s\n",
                    bwc_cuda_mat_device(M), n, k, c);
        bwc_cuda_mat_free(M);
    }
}

/* major index = row of the stored layout */
template<typename Arith>
void matmul_cuda<Arith>::upload()
{
    unsigned int const nmajor = dim[store_transposed];
    unsigned int const nminor = dim[!store_transposed];
    M = bwc_cuda_mat_create(q.data(), q.size(), nmajor, nminor,
            xab->elt_stride() / sizeof(uint64_t));
    if (!M) abort();
    /* the GPU holds the matrix now */
    std::vector<uint32_t>().swap(q);
}

template<typename Arith>
void matmul_cuda<Arith>::build_cache(matrix_u32 && m)
{
    q = std::move(m.p);
}

template<typename Arith>
int matmul_cuda<Arith>::reload_cache_private()
{
    // NOLINTNEXTLINE(readability-static-accessed-through-instance)
    auto f = matmul_common_reload_cache_fopen(xab->elt_stride(), *this, MM_MAGIC);
    if (!f) return 0;

    uint32_t datasize;

    MATMUL_COMMON_READ_ONE32(datasize, f.get());
    resize_and_check_meaningful(q, datasize, f.get());
    MATMUL_COMMON_READ_MANY32(q.data(), datasize, f.get());

    return 1;
}

template<typename Arith>
void matmul_cuda<Arith>::save_cache_private()
{
    ASSERT_ALWAYS(!M);  /* q is dropped once uploaded */
    // NOLINTNEXTLINE(readability-static-accessed-through-instance)
    auto f = matmul_common_save_cache_fopen(xab->elt_stride(), *this, MM_MAGIC);
    if (!f) return;

    MATMUL_COMMON_WRITE_ONE32(q.size(), f.get());
    MATMUL_COMMON_WRITE_MANY32(q.data(), q.size(), f.get());
}

template<typename Arith>
void matmul_cuda<Arith>::mul(void * xdst, void const * xsrc, int d)
{
    if (!M) upload();
    /* d == !store_transposed: dst is indexed by the stored rows */
    bwc_cuda_mat_mul(M, xdst, xsrc, d == !store_transposed);
    iteration[d]++;
}

template<typename Arith>
void matmul_cuda<Arith>::report(double scale MAYBE_UNUSED)
{
    if (!M) return;
    double k, c;
    unsigned long n;
    bwc_cuda_mat_times(M, &k, &c, &n);
    if (!n) return;
    char buf[160];
    snprintf(buf, sizeof(buf), "GPU %d: %.3f ms kernel + %.3f ms copies"
            " per product", bwc_cuda_mat_device(M), 1e3 * k / (double) n,
            1e3 * c / (double) n);
    report_string = buf;
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
matmul_interface * CADO_CONCATENATE4(new_matmul_, ARITH_LAYER, _, MM_IMPL)(
        matmul_public && P,
        arith_generic * arith,
        cxx_param_list & pl,
        int optimized_direction)
{
    return new matmul_cuda<arith_hard>(std::move(P), arith->concrete(), pl, optimized_direction);
}

/* vim: set sw=4: */
