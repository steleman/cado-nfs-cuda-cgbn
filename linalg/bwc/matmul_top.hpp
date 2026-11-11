#ifndef CADO_MATMUL_TOP_HPP
#define CADO_MATMUL_TOP_HPP

#include <cstddef>
#include <cstdint>

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <gmp.h>                 // for gmp_randstate_t

#include "select_mpi.h"
#include "parallelizing_info.hpp"
#include "matmul.hpp"
#include "params.hpp"
#include "balancing.hpp"

struct matmul_top_matrix {
    // global stuff.
    //
    // n[0] is the number of rows, includding padding.
    // n[1] is the number of columns, includding padding.
    //
    // n0[] is without padding.
    //
    // Note that a communicator in direction 0 handles in total a dataset
    // of size n[1] (the number of items in a matrix row is the number of
    // columns.
    unsigned int n[2];
    unsigned int n0[2]; // n0: unpadded.

    // this really ends up within the mm field. It's not a complete file
    // name though. We lack the implementation extension, the possible
    // transposition tag, as well as the .bin extension.
    std::string locfile;

    /* These two are global to all threads and jobs (well, each thread
     * has its own pointer though, it's not a shared_malloc. It could be,
     * but it isn't).
     *
     * For random matrices, both strings below are empty.
     */
    std::string mname;
    std::string bname;

    balancing bal;

    /* These are local excerpts of the balancing permutation: arrays of
     * pairs (row index in the (sub)-matrix) ==> (row index in the
     * original matrix), but only when both coordinates of this pair are
     * in the current row and column range. This can be viewed as the set
     * of non-zero positions in the permutation matrix if it were split
     * just like the current matrix is. */
    std::vector<std::array<unsigned int, 2>> perm[2]; // row,col

    /* XXX XXX XXX dangerous: has_perm_local is only a local thing!
     * Some threads/jobs may have has_perm_local(d)==false while others don't!
     */
    bool has_perm_local(int d) const { return !perm[d].empty(); }

    /* This is the global version, as prepared by
     * matmul_top_init_prepare_local_permutations
     */
    int has_perm_map[2];
    bool has_perm(int d) const { return has_perm_map[d]; }

    /* It's shared because of interleaving.
     * Note that when we get the interface from a shared lib, the shared
     * ptr actually hides a call to a custom deleter.
     */
    std::shared_ptr<matmul_interface> mm;
};

struct matmul_top_data {
    arith_generic * abase;
    parallelizing_info & pi;
    pi_datatype * pitype;
    /* These n[] and n0[] correspond to the dimensions of the product
     *
     * n[0] is matrices[0]->n[0] (number of rows, includding padding.)
     * n[1] is matrices[nmatrices-1]->n[1] (number of columns, includding padding.)
     *
     * and without padding:
     * n0[0] is matrices[0]->n0[0]
     * n0[1] is matrices[nmatrices-1]->n0[1]
     */
    unsigned int n[2] { 0, 0 };
    unsigned int n0[2] { 0, 0 }; // n0: unpadded.

    /* The matrix we are dealing with is
     * matrices[0] * matrices[1] * ... * matrices[nmatrices-1]
     */
    std::vector<matmul_top_matrix> matrices;

    /* mm_bucket_heatmap. When set, matmul_top_report collects the heat
     * maps of all the submatrices into this one file. Empty by default,
     * and then nothing of the sort happens. */
    std::string heatmap_file;

    matmul_top_data(
        arith_generic * abase,
        parallelizing_info & pi,
        cxx_param_list & pl,
        int optimized_direction);
    ~matmul_top_data();
};

struct mmt_vec;


extern void matmul_top_decl_usage(cxx_param_list & pl);
extern void matmul_top_lookup_parameters(cxx_param_list & pl);
extern void matmul_top_report(matmul_top_data & mmt, double scale, int full);
extern unsigned int matmul_top_rank_upper_bound(matmul_top_data & mmt);
#if 0
extern void matmul_top_fill_random_source(matmul_top_data & mmt, int d);
#endif
extern void mmt_vec_truncate(matmul_top_data & mmt, mmt_vec & v);
extern void mmt_vec_truncate_above_index(matmul_top_data & mmt, mmt_vec & v, unsigned int idx);
extern void mmt_vec_truncate_below_index(matmul_top_data & mmt, mmt_vec & v, unsigned int idx);
extern void matmul_top_mul_cpu(matmul_top_data & mmt, int midx, int d, mmt_vec & w, mmt_vec const & v);
extern void matmul_top_comm_bench(matmul_top_data & mmt, int d);
extern void matmul_top_mul_comm(mmt_vec & w, mmt_vec & v);

/* v is both input and output. w is temporary */
extern void matmul_top_mul(matmul_top_data & mmt, mmt_vec *w, struct timing_data * tt);

extern void mmt_vec_twist(matmul_top_data & mmt, mmt_vec & y);
extern void mmt_vec_untwist(matmul_top_data & mmt, mmt_vec & y);
extern void mmt_vec_apply_T(matmul_top_data & mmt, mmt_vec & y);
extern void mmt_vec_unapply_T(matmul_top_data & mmt, mmt_vec & y);
extern void mmt_vec_apply_S(matmul_top_data & mmt, int midx, mmt_vec & y);
extern void mmt_vec_unapply_S(matmul_top_data & mmt, int midx, mmt_vec & y);
extern void mmt_vec_apply_P(matmul_top_data & mmt, mmt_vec & y);
extern void mmt_vec_unapply_P(matmul_top_data & mmt, mmt_vec & y);
extern void mmt_apply_identity(mmt_vec & w, mmt_vec const & v);
extern void indices_twist(matmul_top_data & mmt, std::vector<uint32_t> & xs, int d);

/* {{{ products with the vectors kept on the GPUs (mm_impl=cuda)
 *
 * Usable (operator bool) with one thread per MPI job, no interleaving,
 * one matrix and the cuda backend; the caller keeps its host loop
 * otherwise, or with CADO_BWC_CUDA_RESIDENT=0. Each job's matrix block
 * is on its own GPU.
 *
 * A device vector holds the job's whole slice [i0,i1) of a vector in
 * direction d (that of v0), the same on every job sharing the slice:
 * an mmt_vec with consistency 2.
 *
 * mul(cur, tmp) is matmul_top_mul with its communication; the result is
 * in cur afterwards, tmp is scratch (the two may be swapped):
 *  - one job: tmp = M cur and a swap, since reduce + broadcast is then
 *    a copy;
 *  - several: tmp = M_local cur; an XOR reduce-scatter over the jobs of
 *    pi.wr[!d] (chunk k of tmp to job k), whose result is our piece of
 *    cur; then an in-place all-gather of the pieces over pi.wr[d]. That
 *    is what mmt_vec_reduce + mmt_vec_broadcast do with one thread per
 *    job. The exchanges go through NCCL, or through MPI with host
 *    buffers (CADO_BWC_GPU_COMM=mpi; the only choice without NCCL, and
 *    the one that works with several jobs on one GPU).
 */
struct bwc_cuda_mat;
struct mmt_cuda_transport;
class mmt_cuda {
    matmul_top_data & mmt;
    int d;
    bwc_cuda_mat * G = nullptr;
    int gather = 0;
    size_t elt = 0;             /* bytes per item */
    size_t eblock = 0;          /* items in a job's own piece */
    size_t own_off = 0;         /* items: our piece within the slice */
    void * slots = nullptr;     /* reduce-scatter receive buffers */
    std::unique_ptr<mmt_cuda_transport> tr;
  public:
    size_t vitems = 0;          /* items in the slice */
    size_t vbytes = 0;
    /* size for both buffers given to mul() (the product's output slice
     * may be larger than the input one) */
    size_t tmpbytes = 0;
    mmt_cuda(matmul_top_data & mmt, mmt_vec const & v0, mmt_vec const & v1);
    ~mmt_cuda();
    mmt_cuda(mmt_cuda const &) = delete;
    mmt_cuda & operator=(mmt_cuda const &) = delete;
    explicit operator bool() const { return G != nullptr; }
    bwc_cuda_mat * mat() const { return G; }
    /* "local", "nccl" or "mpi" */
    const char * comm_name() const;
    void mul(void *& cur, void *& tmp);
};
/* }}} */

#endif	/* MATMUL_TOP_HPP_ */
