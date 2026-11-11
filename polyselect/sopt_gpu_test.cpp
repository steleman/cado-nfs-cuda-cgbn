/* Checks the GPU size optimization against the CPU code.
 *
 *   sopt_gpu_test <poly file> [variants per pair] [seed]
 *
 * The file holds polynomial pairs in CADO format (c0..cd, Y0, Y1), such
 * as polyselect or polyselect_ropt output. Each pair and random rotations
 * and translations of it are size-optimized: L2_skewness / L2_skew_lognorm
 * (the GPU's tt finished on the CPU must give the CPU's value, bit for
 * bit) and sopt_local_descent (same polynomials), with timings.
 */

#include "cado.h" // IWYU pragma: keep

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "mpz_poly.h"
#include "polyselect_norms.hpp"
#include "size_optimization.hpp"
#include "sopt_gpu_kernels.h"
#include "sopt_gpu.hpp"

static double now()
{
    return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* two's complement, nl limbs; false if it does not fit */
static bool to_limbs(uint64_t * w, mpz_srcptr z, int nl)
{
    if (mpz_sizeinbase(z, 2) >= (size_t) 64 * nl - 4) return false;
    cxx_mpz a;
    mpz_abs(a, z);
    memset(w, 0, nl * sizeof(uint64_t));
    size_t cnt;
    mpz_export(w, &cnt, -1, sizeof(uint64_t), 0, 0, a);
    if (mpz_sgn(z) < 0) {
        unsigned c = 1;
        for (int i = 0; i < nl; i++) {
            w[i] = ~w[i] + c;
            c = c && w[i] == 0;
        }
    }
    return true;
}

int main(int argc, char ** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <poly file> [variants] [seed]\n", argv[0]);
        return 1;
    }
    const int variants = argc > 2 ? atoi(argv[2]) : 20;
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, argc > 3 ? atoi(argv[3]) : 1);

    /* read the pairs */
    std::vector<cxx_mpz_poly> F, G;
    {
        std::ifstream in(argv[1]);
        std::string line;
        cxx_mpz_poly f, g;
        int have = 0;
        while (std::getline(in, line)) {
            if (line.size() > 2 && line[0] == 'c' && isdigit(line[1])) {
                int i = atoi(line.c_str() + 1);
                cxx_mpz z;
                mpz_set_str(z, line.c_str() + line.find(':') + 1 + (line[line.find(':') + 1] == ' '), 10);
                mpz_poly_setcoeff(f, i, z);
                have |= 1;
            } else if (line.rfind("Y0:", 0) == 0 || line.rfind("Y1:", 0) == 0) {
                cxx_mpz z;
                mpz_set_str(z, line.c_str() + 4, 10);
                mpz_poly_setcoeff(g, line[1] - '0', z);
                have |= line[1] == '0' ? 2 : 4;
            }
            if (have == 7) {
                F.push_back(f);
                G.push_back(g);
                f = cxx_mpz_poly();
                g = cxx_mpz_poly();
                have = 0;
            }
        }
    }
    if (F.empty()) {
        fprintf(stderr, "no polynomial pairs in %s\n", argv[1]);
        return 1;
    }
    const int d = F[0]->deg;

    /* variants: f + k x^i g for random i and k (up to 2^40), then a
     * translation by a random k (up to 2^12) */
    std::vector<cxx_mpz_poly> VF, VG;
    for (size_t j = 0; j < F.size(); j++) {
        VF.push_back(F[j]);
        VG.push_back(G[j]);
        for (int v = 0; v < variants; v++) {
            cxx_mpz k;
            cxx_mpz_poly f, g;
            mpz_urandomb(k, rs, 1 + gmp_urandomm_ui(rs, 40));
            if (gmp_urandomm_ui(rs, 2)) mpz_neg(k, k);
            mpz_poly_rotation(f, F[j], G[j], k, gmp_urandomm_ui(rs, d - 1));
            mpz_urandomb(k, rs, 1 + gmp_urandomm_ui(rs, 12));
            if (gmp_urandomm_ui(rs, 2)) mpz_neg(k, k);
            mpz_poly_translation(f, f, k);
            mpz_poly_translation(g, G[j], k);
            VF.push_back(f);
            VG.push_back(g);
        }
    }
    const int n = VF.size();
    size_t maxbits = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= d; i++)
            maxbits = std::max(maxbits, mpz_sizeinbase(mpz_poly_coeff_const(VF[j], i), 2));
    int nl = 2;
    while (64 * nl < (int) maxbits + 96) nl *= 2;
    printf("%zu pairs, %d polynomials of degree %d, coefficients up to %zu bits: %d limbs\n",
            F.size(), n, d, maxbits, nl);

    struct ssg_ctx * ctx = ssg_ctx_create();
    if (!ctx) {
        fprintf(stderr, "no GPU\n");
        return 1;
    }

    /* 1. L2_skewness, L2_skew_lognorm */
    {
        std::vector<uint64_t> limbs((size_t) n * (d + 1) * nl);
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= d; i++)
                to_limbs(&limbs[((size_t) j * (d + 1) + i) * nl], mpz_poly_coeff_const(VF[j], i), nl);
        std::vector<double> skew(n), tt(n), ap(n);
        std::vector<int> fix(n), st(n);
        ssg_l2_test(ctx, n, d, nl, limbs.data(), skew.data(), tt.data(), fix.data(), ap.data(), st.data());
        double t0 = now();
        ssg_l2_test(ctx, n, d, nl, limbs.data(), skew.data(), tt.data(), fix.data(), ap.data(), st.data());
        const double tg = now() - t0;
        int bad_skew = 0, bad_norm = 0, notok = 0;
        double maxdiff = 0;
        t0 = now();
        for (int j = 0; j < n; j++) {
            const double s = L2_skewness(VF[j]);
            const double l = L2_skew_lognorm(VF[j]);
            if (st[j] != SSG_OK) { notok++; continue; }
            if (s != skew[j]) bad_skew++;
            const double l2 = std::fma((double) fix[j], std::numbers::ln2_v<double> / 2, std::log(tt[j]) / 2);
            if (l2 != l) bad_norm++;
            maxdiff = std::max(maxdiff, std::fabs(ap[j] - l));
        }
        const double tc = now() - t0;
        printf("L2: %d polynomials, %d not done on the GPU; skewness differs %d, lognorm (GPU tt, CPU log) differs %d;"
               " GPU log off by at most %.3g\n", n, notok, bad_skew, bad_norm, maxdiff);
        printf("L2: GPU %.3f s (%.1f ns each), CPU %.3f s for skewness + lognorm (%.0f ns each)\n",
                tg, tg / n * 1e9, tc, tc / n * 1e9);
    }

    /* 2. local descents, as size_optimization_aux calls them, through
     * sopt_gpu_descents (GPU, then the CPU finishes or redoes what it gives
     * back): same polynomials and return values as sopt_local_descent */
    {
        const int degrot = d - 2;
        setenv("CADO_POLYSELECT_GPU_SOPT", "1", 1);
        std::vector<cxx_mpz_poly> GF = VF, GG = VG;
        std::vector<double> GL;
        double t0 = now();
        if (!sopt_gpu_descents(GF, GG, degrot, SOPT_DEFAULT_MAX_STEPS, GL)) {
            fprintf(stderr, "sopt_gpu_descents did not use the GPU\n");
            return 1;
        }
        const double tg = now() - t0;
        const int ncpu = getenv("SOPT_TEST_NOCPU") ? std::min(n, 2000) : n;
        int bad = 0;
        t0 = now();
        for (int j = 0; j < ncpu; j++) {
            cxx_mpz_poly fo, go;
            const double l = sopt_local_descent(fo, go, VF[j], VG[j], 1, degrot, SOPT_DEFAULT_MAX_STEPS, 0);
            if (mpz_poly_cmp(GF[j], fo) || mpz_poly_cmp(GG[j], go) || GL[j] != l)
                bad++;
        }
        const double tc = now() - t0;
        printf("descent: %d through sopt_gpu_descents, %d compared with sopt_local_descent: %d differ\n",
                n, ncpu, bad);
        printf("descent: GPU (+ CPU for the ones given back) %.3f s, CPU %.3f s (one thread, %d descents)\n",
                tg, tc, ncpu);
        if (bad) return 1;
    }
    ssg_ctx_free(ctx);
    return 0;
}
