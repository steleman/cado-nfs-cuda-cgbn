#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gmp.h>

#include "macros.h"
#include "polyselect_norms.hpp"
#include "size_optimization.hpp"
#include "sopt_gpu.hpp"

#ifdef HAVE_CUDA
#include "sopt_gpu_kernels.h"

namespace {
struct sopt_gpu_holder {
    struct ssg_ctx * ctx = nullptr;
    bool tried = false;
    bool use = false;
    unsigned long cpu_resumed = 0, cpu_redone = 0;
    ~sopt_gpu_holder () {
        if (!ctx)
            return;
        if (getenv ("CADO_POLYSELECT_GPU_VERBOSE") && use) {
            double sec;
            unsigned long n, back;
            ssg_ctx_stats (ctx, &sec, &n, &back);
            fprintf (stderr, "# polyselect GPU size optimization: %lu descents"
                     " in %.2f s; the CPU finished %lu and redid %lu\n",
                     n, sec, cpu_resumed, cpu_redone);
        }
        ssg_ctx_free (ctx);
    }
};
}

static sopt_gpu_holder * sopt_gpu_holder_get ()
{
    static thread_local sopt_gpu_holder h;
    if (!h.tried) {
        h.tried = true;
        const char * all = getenv ("CADO_POLYSELECT_GPU");
        const char * e = getenv ("CADO_POLYSELECT_GPU_SOPT");
        if ((all && all[0] == '0') || (e && e[0] == '0'))
            return &h;
        h.ctx = ssg_ctx_create ();
        if (!h.ctx)
            return &h;
        const int ratio = ssg_fp64_ratio (h.ctx);
        /* by default only where double precision is fast */
        h.use = (e && e[0] == '1') || (ratio > 0 && ratio <= 4);
    }
    return &h;
}

bool sopt_gpu_enabled ()
{
    return sopt_gpu_holder_get ()->use;
}

/* two's complement, nl limbs (the caller checks the size) */
static void to_limbs (uint64_t * w, mpz_srcptr z, int nl)
{
    cxx_mpz a;
    mpz_abs (a, z);
    memset (w, 0, nl * sizeof (uint64_t));
    size_t cnt;
    mpz_export (w, &cnt, -1, sizeof (uint64_t), 0, 0, a);
    if (mpz_sgn (z) < 0) {
        unsigned c = 1;
        for (int i = 0; i < nl; i++) {
            w[i] = ~w[i] + c;
            c = c && w[i] == 0;
        }
    }
}

static void from_limbs (mpz_ptr z, const uint64_t * w, int nl)
{
    const bool neg = (int64_t) w[nl - 1] < 0;
    std::vector<uint64_t> a (w, w + nl);
    if (neg) {
        unsigned c = 1;
        for (int i = 0; i < nl; i++) {
            a[i] = ~a[i] + c;
            c = c && a[i] == 0;
        }
    }
    mpz_import (z, nl, -1, sizeof (uint64_t), 0, 0, a.data ());
    if (neg)
        mpz_neg (z, z);
}

bool sopt_gpu_descents (std::vector<cxx_mpz_poly> & f,
                        std::vector<cxx_mpz_poly> & g, int deg_rotation,
                        unsigned int max_iter, std::vector<double> & lognorm)
{
    sopt_gpu_holder * h = sopt_gpu_holder_get ();
    if (!h->use)
        return false;
    const size_t n = f.size ();
    lognorm.assign (n, 0);
    if (n == 0)
        return true;
    const int d = f[0]->deg;
    size_t bits = 0;
    bool ok = d >= 2 && d <= 8 && deg_rotation <= d - 2 && deg_rotation <= 7;
    for (size_t i = 0; ok && i < n; i++) {
        ok = f[i]->deg == d && g[i]->deg == 1;
        for (int j = 0; ok && j <= d; j++)
            bits = std::max (bits, mpz_sizeinbase (mpz_poly_coeff_const (f[i], j), 2));
        for (int j = 0; ok && j <= 1; j++)
            bits = std::max (bits, mpz_sizeinbase (mpz_poly_coeff_const (g[i], j), 2));
    }
    /* room for the translations and rotations of the descent; more than
       that and the GPU gives the descent back */
    int nl = 2;
    while (64 * nl < (int) bits + 96 && nl < 16)
        nl *= 2;
    if (!ok || 64 * nl < (int) bits + 96)
        return false;

    std::vector<uint64_t> fl (n * (d + 1) * nl), gl (n * 2 * nl);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j <= d; j++)
            to_limbs (&fl[(i * (d + 1) + j) * nl], mpz_poly_coeff_const (f[i], j), nl);
        for (int j = 0; j < 2; j++)
            to_limbs (&gl[(i * 2 + j) * nl], mpz_poly_coeff_const (g[i], j), nl);
    }
    std::vector<int> st (n), state (n * SSG_STATE_INTS);
    ssg_descent (h->ctx, (int) n, d, nl, fl.data (), gl.data (), deg_rotation,
                 max_iter, st.data (), state.data ());

    for (size_t i = 0; i < n; i++) {
        if (st[i] == SSG_OK || st[i] == SSG_RESUME) {
            cxx_mpz_poly fo, go;
            for (int j = 0; j <= d; j++) {
                cxx_mpz z;
                from_limbs (z, &fl[(i * (d + 1) + j) * nl], nl);
                mpz_poly_setcoeff (fo, j, z);
            }
            for (int j = 0; j < 2; j++) {
                cxx_mpz z;
                from_limbs (z, &gl[(i * 2 + j) * nl], nl);
                mpz_poly_setcoeff (go, j, z);
            }
            if (st[i] == SSG_OK) {
                /* sopt_local_descent returns L2_skew_lognorm (f_opt) */
                lognorm[i] = L2_skew_lognorm (fo);
            } else {
                const int * s = &state[i * SSG_STATE_INTS];
                lognorm[i] = sopt_local_descent_continue (fo, go, s[1], s + 2,
                        (unsigned int) s[0], 1, deg_rotation, max_iter, 0);
                h->cpu_resumed++;
            }
            f[i] = fo;
            g[i] = go;
        } else {
            cxx_mpz_poly fo, go;
            lognorm[i] = sopt_local_descent (fo, go, f[i], g[i], 1, deg_rotation,
                                             max_iter, 0);
            f[i] = fo;
            g[i] = go;
            h->cpu_redone++;
        }
    }
    return true;
}

#else

bool sopt_gpu_enabled () { return false; }

bool sopt_gpu_descents (std::vector<cxx_mpz_poly> &, std::vector<cxx_mpz_poly> &,
                        int, unsigned int, std::vector<double> &)
{
    return false;
}

#endif
