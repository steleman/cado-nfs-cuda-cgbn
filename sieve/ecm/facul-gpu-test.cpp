#include "cado.h" // IWYU pragma: keep

/* Compares the GPU cofactorization with facul_all on a file of
 * candidates as las -dump-cofactors writes them ("a b n0 n1" per line).
 *
 *   facul_gpu_test -lim0 L0 -lim1 L1 -lpb0 .. -lpb1 .. -mfb0 .. -mfb1 ..
 *                  -ncurves0 .. -ncurves1 .. [-t threads] [-batch n] file
 *
 * Prints the time of both, and every candidate whose outcome differs
 * (status, or the primes of a smooth one).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "facul-gpu.hpp"
#include "facul.hpp"
#include "facul_strategies.hpp"
#include "timing.h"

/* factor_leftover_norms of las-cofactor.cpp */
static int cpu_factor(std::vector<cxx_mpz> const & n,
        std::vector<std::vector<cxx_mpz>> & pr, facul_strategies const & strat)
{
    auto fac = facul_all(n, strat);
    for (auto const & f : fac)
        if (f.status == FACUL_NOT_SMOOTH) return -1;
    for (size_t s = 0; s < fac.size(); s++) {
        if (fac[s].status == FACUL_MAYBE) return 0;
        std::swap(pr[s], fac[s].primes);
    }
    return 1;
}

static void usage()
{
    fprintf(stderr, "usage: facul_gpu_test -lim0 L -lim1 L -lpb0 l -lpb1 l"
            " -mfb0 m -mfb1 m [-ncurves0 n] [-ncurves1 n] [-t threads]"
            " [-batch n] [-nocpu] [-notperfect] file\n");
    exit(1);
}

int main(int argc, char ** argv)
{
    std::vector<unsigned long> lim(2, 0);
    std::vector<unsigned int> lpb(2, 0), mfb(2, 0);
    std::vector<int> ncurves(2, -1);
    int threads = 1;
    size_t batch = 1 << 20;
    bool nocpu = false;
    bool perfect = true;
    const char * file = nullptr;
    for (int i = 1; i < argc; i++) {
        std::string const a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-lim0") lim[0] = strtoul(next(), nullptr, 10);
        else if (a == "-lim1") lim[1] = strtoul(next(), nullptr, 10);
        else if (a == "-lpb0") lpb[0] = atoi(next());
        else if (a == "-lpb1") lpb[1] = atoi(next());
        else if (a == "-mfb0") mfb[0] = atoi(next());
        else if (a == "-mfb1") mfb[1] = atoi(next());
        else if (a == "-ncurves0") ncurves[0] = atoi(next());
        else if (a == "-ncurves1") ncurves[1] = atoi(next());
        else if (a == "-t") threads = atoi(next());
        else if (a == "-batch") batch = strtoul(next(), nullptr, 10);
        else if (a == "-nocpu") nocpu = true;
        /* as las builds its strategies (BB = 0, see facul_make_strategies) */
        else if (a == "-notperfect") perfect = false;
        else if (a[0] == '-') usage();
        else file = argv[i];
    }
    if (!file || !lim[0] || !lim[1]) usage();

    facul_strategies const strat(lim, lpb, mfb, ncurves, perfect, 0);

    std::vector<std::vector<cxx_mpz>> N;
    {
        FILE * f = fopen(file, "r");
        if (!f) { perror(file); return 1; }
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            char * p = line;
            strtoll(p, &p, 10);     /* a */
            strtoull(p, &p, 10);    /* b */
            std::vector<cxx_mpz> n(2);
            for (auto & x : n) {
                while (*p == ' ') p++;
                char * e = p;
                while (*e >= '0' && *e <= '9') e++;
                char const c = *e;
                *e = 0;
                mpz_set_str(x, p, 10);
                *e = c;
                p = e;
            }
            N.push_back(n);
        }
        fclose(f);
    }
    size_t const nc = N.size();
    printf("%zu candidates\n", nc);

    facul_gpu * g = facul_gpu::get();
    if (!g) { fprintf(stderr, "no GPU\n"); return 1; }

    /* GPU */
    std::vector<facul_gpu_outcome> gout(nc);
    double t0 = wct_seconds();
    std::vector<std::vector<cxx_mpz> const *> ptrs;
    std::vector<facul_gpu_outcome> part;
    for (size_t c0 = 0; c0 < nc; c0 += batch) {
        size_t const c1 = std::min(nc, c0 + batch);
        ptrs.clear();
        for (size_t c = c0; c < c1; c++) ptrs.push_back(&N[c]);
        g->factor_all(ptrs, strat, part);
        for (size_t c = c0; c < c1; c++) gout[c] = std::move(part[c - c0]);
    }
    double const tg = wct_seconds() - t0;
    double ks;
    unsigned long nj;
    g->stats(ks, nj);
    size_t ncpu = 0, nsmooth_g = 0;
    for (auto const & o : gout) {
        ncpu += o.cpu;
        nsmooth_g += !o.cpu && o.status == 1;
    }
    printf("GPU: %.3f s wall, kernels %.3f s, %lu jobs, %zu left to the CPU,"
            " %zu smooth\n", tg, ks, nj, ncpu, nsmooth_g);
    if (nocpu) return 0;

    /* CPU */
    std::vector<int> cst(nc);
    std::vector<std::vector<std::vector<cxx_mpz>>> cpr(nc);
    std::atomic<size_t> cur(0);
    t0 = wct_seconds();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++)
        th.emplace_back([&]() {
            for (size_t c; (c = cur++) < nc;) {
                std::vector<std::vector<cxx_mpz>> pr(2);
                cst[c] = cpu_factor(N[c], pr, strat);
                cpr[c] = pr;
            }
        });
    for (auto & t : th) t.join();
    double const tc = wct_seconds() - t0;
    size_t nsmooth_c = 0;
    for (int s : cst) nsmooth_c += s == 1;
    printf("CPU: %.3f s wall on %d threads, %zu smooth\n", tc, threads, nsmooth_c);

    /* compare */
    size_t diff_status = 0, diff_primes = 0, maybe_vs_not = 0;
    for (size_t c = 0; c < nc; c++) {
        auto const & o = gout[c];
        if (o.cpu) continue;
        int const cs = cst[c];
        if (o.status != cs) {
            /* the GPU does the sides one after the other: where the CPU
             * finds side 1 not smooth after a split on side 0 that the
             * GPU could not finish, the GPU says "maybe". Both reject. */
            if (o.status == 0 && cs == -1) { maybe_vs_not++; continue; }
            if (diff_status < 20)
                gmp_printf("status %d (GPU) %d (CPU): %Zd %Zd\n", o.status, cs,
                        (mpz_srcptr) N[c][0], (mpz_srcptr) N[c][1]);
            diff_status++;
            continue;
        }
        if (cs == 1 && o.primes != cpr[c]) {
            if (diff_primes < 20)
                gmp_printf("primes differ: %Zd %Zd\n",
                        (mpz_srcptr) N[c][0], (mpz_srcptr) N[c][1]);
            diff_primes++;
        }
    }
    printf("differences: %zu status, %zu primes (%zu maybe vs not smooth,"
            " same verdict)\n", diff_status, diff_primes, maybe_vs_not);
    return 0;
}
