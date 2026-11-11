/* sqrt_gpu_mul_test: GPU products against mpz_mul, with timings */
#include "cado.h" // IWYU pragma: keep
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <gmp.h>
#include "sqrt_gpu_mul.h"

static double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv)
{
    if (!sgm_init()) { fprintf(stderr, "no GPU\n"); return 1; }
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, argc > 1 ? atoi(argv[1]) : 1);
    int bad = 0;
    /* correctness: many sizes, all-ones operands, squares */
    for (int t = 0; t < 400; t++) {
        const size_t na = 1 + gmp_urandomm_ui(rs, t < 300 ? 3000 : 200000);
        const size_t nb = (t % 5 == 0) ? na : 1 + gmp_urandomm_ui(rs, na);
        mpz_t a, b, r;
        mpz_inits(a, b, r, NULL);
        if (t % 7 == 0) {
            mpz_set_ui(a, 1); mpz_mul_2exp(a, a, 64 * na); mpz_sub_ui(a, a, 1);
            mpz_set_ui(b, 1); mpz_mul_2exp(b, b, 64 * nb); mpz_sub_ui(b, b, 1);
        } else {
            mpz_urandomb(a, rs, 64 * na);
            mpz_urandomb(b, rs, 64 * nb);
            mpz_setbit(a, 64 * na - 1);
            mpz_setbit(b, 64 * nb - 1);
        }
        const bool sq = t % 5 == 0;
        if (sq) mpz_set(b, a);
        mpz_mul(r, a, b);
        std::vector<uint64_t> out(na + nb);
        const uint64_t * pa = (const uint64_t *) mpz_limbs_read(a);
        sgm_mul(out.data(), pa, na, sq ? pa : (const uint64_t *) mpz_limbs_read(b), nb);
        for (size_t i = 0; i < na + nb; i++)
            if (out[i] != (i < mpz_size(r) ? mpz_getlimbn(r, i) : 0)) { bad++; break; }
        mpz_clears(a, b, r, NULL);
    }
    printf("400 products (up to 200000 limbs): %d wrong\n", bad);
    /* timings */
    for (size_t n = 1024; n <= (1u << 20); n *= 2) {
        mpz_t a, b, r;
        mpz_inits(a, b, r, NULL);
        mpz_urandomb(a, rs, 64 * n);
        mpz_urandomb(b, rs, 64 * n);
        std::vector<uint64_t> out(2 * n);
        double t0 = now();
        int k = 0;
        do { mpz_mul(r, a, b); k++; } while (now() - t0 < 0.3);
        const double tc = (now() - t0) / k;
        sgm_mul(out.data(), (const uint64_t *) mpz_limbs_read(a), n, (const uint64_t *) mpz_limbs_read(b), n);
        t0 = now();
        k = 0;
        do { sgm_mul(out.data(), (const uint64_t *) mpz_limbs_read(a), n, (const uint64_t *) mpz_limbs_read(b), n); k++; } while (now() - t0 < 0.3);
        const double tg = (now() - t0) / k;
        printf("%8zu limbs (%9zu bits): GMP %9.3f ms, GPU %8.3f ms\n", n, 64 * n, tc * 1e3, tg * 1e3);
        mpz_clears(a, b, r, NULL);
    }
    return bad != 0;
}
