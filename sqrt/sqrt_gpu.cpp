#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <gmp.h>

#include "gmp_mul_hook.hpp"
#include "macros.h"
#include "sqrt_gpu.hpp"

#ifdef HAVE_CUDA
#include "sqrt_gpu_mul.h"

static bool sqrt_gpu_check = false;

static void sqrt_gpu_mpz_mul (mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
    const size_t na = mpz_size (a), nb = mpz_size (b);
    const int sign = mpz_sgn (a) * mpz_sgn (b);
    /* r may be a or b */
    std::vector<uint64_t> out (na + nb);
    const uint64_t * pa = (const uint64_t *) mpz_limbs_read (a);
    const uint64_t * pb = (a == b) ? pa : (const uint64_t *) mpz_limbs_read (b);
    sgm_mul (out.data (), pa, na, pb, nb);
    size_t n = na + nb;
    while (n && out[n - 1] == 0)
        n--;
    if (sqrt_gpu_check) {
        mpz_t ref;
        mpz_init (ref);
        mpz_mul (ref, a, b);
        mpz_t got;
        mpz_roinit_n (got, (const mp_limb_t *) out.data (), sign < 0 ? -(mp_size_t) n : (mp_size_t) n);
        if (mpz_cmp (ref, got) != 0) {
            fprintf (stderr, "sqrt GPU check: wrong product of %zu x %zu limbs\n", na, nb);
            abort ();
        }
        mpz_clear (ref);
    }
    mp_limb_t * pr = mpz_limbs_write (r, (mp_size_t) (n ? n : 1));
    for (size_t i = 0; i < n; i++)
        pr[i] = out[i];
    mpz_limbs_finish (r, sign < 0 ? -(mp_size_t) n : (mp_size_t) n);
}

void sqrt_gpu_init ()
{
    static_assert (sizeof (mp_limb_t) == sizeof (uint64_t), "64-bit limbs");
    const char * e = getenv ("CADO_SQRT_GPU");
    if (e && e[0] == '0')
        return;
    if (!sgm_init ())
        return;
    sqrt_gpu_check = getenv ("CADO_SQRT_GPU_CHECK") != nullptr;
    const char * m = getenv ("CADO_SQRT_GPU_MIN");
    cado_mpz_mul_hook_threshold = m ? strtoul (m, nullptr, 10) : 32768;
    cado_mpz_mul_hook = sqrt_gpu_mpz_mul;
}

void sqrt_gpu_report ()
{
    if (!cado_mpz_mul_hook || !getenv ("CADO_SQRT_GPU_VERBOSE"))
        return;
    double sec;
    unsigned long calls;
    sgm_stats (&sec, &calls);
    fprintf (stderr, "# sqrt GPU: %lu products of >= %zu limbs in %.2f s"
             " on %d GPU(s)\n",
             calls, cado_mpz_mul_hook_threshold, sec, sgm_ngpu ());
}

#else

void sqrt_gpu_init () {}
void sqrt_gpu_report () {}

#endif
