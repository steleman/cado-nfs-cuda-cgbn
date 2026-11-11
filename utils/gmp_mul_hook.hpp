#ifndef CADO_UTILS_GMP_MUL_HOOK_HPP
#define CADO_UTILS_GMP_MUL_HOOK_HPP

/* An optional replacement for mpz_mul on large operands (the square
 * root installs a GPU multiplication there, see sqrt/sqrt_gpu.cpp).
 * Files that want it include this header after gmp.h and use
 * cado_mpz_mul, or redirect mpz_mul to it. The replacement must compute
 * the same product (it is an integer: any correct algorithm does). */

#include <cstddef>
#include <gmp.h>

using cado_mpz_mul_fn = void (*)(mpz_ptr, mpz_srcptr, mpz_srcptr);

/* NULL: always GMP. Set once, before any thread uses it. */
extern cado_mpz_mul_fn cado_mpz_mul_hook;
/* used when both operands have at least this many limbs */
extern size_t cado_mpz_mul_hook_threshold;

static inline void cado_mpz_mul(mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
    if (cado_mpz_mul_hook && mpz_size(a) >= cado_mpz_mul_hook_threshold
            && mpz_size(b) >= cado_mpz_mul_hook_threshold)
        cado_mpz_mul_hook(r, a, b);
    else
        mpz_mul(r, a, b);
}

#endif
