#ifndef CADO_CHARACTERS_GPU_H
#define CADO_CHARACTERS_GPU_H

/* The character matrix of `characters` on the GPU (characters_gpu.cu).
 *
 * For each pair (a, b) and each character (as in characters.cpp,
 * eval_64chars):
 *   p != 0:           the Legendre symbol of b r - a modulo p, bit set if
 *                     it is -1 (0 is an error: the row is flagged);
 *   p = 0, r = 0:     0 (padding); r = 1: 1 (parity);
 *   p = 0, r = 3:     b == 0;
 *   p = 0, r = 2 or 1024 + side: the sign of F(a, b) = sum f_k a^k b^(d-k)
 *                     on that side, bit set if negative, computed exactly
 *                     with CGBN.
 * The Legendre symbols use 64-bit Montgomery arithmetic; both are exact,
 * so the bits are the CPU's.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct chg_char { uint64_t p, r; };

/* a polynomial for the sign characters: deg + 1 coefficients, each the
 * absolute value in nlimbs 32-bit limbs (least significant first) and a
 * sign (-1, 0, 1) */
struct chg_poly {
    int deg;
    int nlimbs;
    const uint32_t * limbs;     /* (deg + 1) * nlimbs */
    const int8_t * sign;        /* deg + 1 */
};

/* 1 if there is a usable GPU */
int chg_available(void);

/* out[i * (nchars / 64) + w] = bits of characters 64 w .. 64 w + 63 for
 * pair i; bad[i] = 1 if a Legendre symbol was 0 (the caller reports it).
 * polys[side] for the sides with sign characters (others may be
 * deg < 0). Returns 0 (nothing done) if the values do not fit. */
int chg_eval(const int64_t * a, const uint64_t * b, size_t n,
        const struct chg_char * chars, int nchars,
        const struct chg_poly * polys, int nsides,
        uint64_t * out, uint8_t * bad);

#ifdef __cplusplus
}
#endif

#endif
