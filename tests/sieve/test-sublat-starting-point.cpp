#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <random>

#include "cxx_mpz.hpp"
#include "gmp_aux.h"
#include "las-plattice.hpp"
#include "las-sublat.hpp"
#include "params.hpp"

/* Checks plattice_enumerator_base::starting_point() against a direct
 * search. Under a sublattice of modulus M, class (si, sj) consists of
 * the (i, j) = (M*i'+si, M*j'+sj), and the first point of the lattice
 * i = r*j mod q in that class is the one with j' >= 0 and
 * -I/2 <= i' < I/2 that is smallest in the order of
 * x = (i'+I/2) + (j' << logI), which is the order of the enumeration. */

static uint64_t first_point(uint64_t q, uint64_t r, int logI, uint32_t M,
        uint32_t si, uint32_t sj)
{
    int64_t const I = int64_t(1) << logI;
    int64_t const Q = q;
    /* q < 2^21, so the products below fit */
    int64_t const invM = int64_t(uint64_t(cxx_mpz(M).invmod(q)));
    for (int64_t jp = 0; jp < 4 * Q; jp++) {
        int64_t const j = M * jp + sj;
        /* M*i' = r*j - si mod q */
        int64_t const ip_mod = ((int64_t(r) * (j % Q) - si) % Q + Q) % Q * invM % Q;
        /* the smallest i' >= -I/2 in that class mod q */
        int64_t const ip = -I / 2 + ((ip_mod + I / 2) % Q);
        if (ip < I / 2)
            return uint64_t(ip + I / 2) + (uint64_t(jp) << logI);
    }
    return UINT64_MAX;
}

template <uint32_t M>
static unsigned long check(std::mt19937_64 & rng, unsigned long ntests,
        unsigned long & compared)
{
    unsigned long bad = 0;
    for (unsigned long k = 0; k < ntests; k++) {
        int const logI = 8 + int(rng() % 5);
        uint64_t q;
        do {
            q = (uint64_t(1) << logI) + rng() % (uint64_t(1) << 20);
        } while (!ulong_isprime(q));
        uint64_t const r = rng() % q;
        plattice_info const pli(q, r, false, logI);
        if (pli.is_discarded())
            continue;
        for (uint32_t si = 0; si < M; si++) {
            for (uint32_t sj = 0; sj < M; sj++) {
                if (si == 0 && sj == 0)
                    continue;
                sublat_t<M> const s(sublat_runtime_t(M, si, sj));
                plattice_x_t const x =
                    plattice_enumerator_base::starting_point(pli, logI, s);
                /* The degenerate bases for which starting_point() gives
                 * up are not checked */
                if (x == plattice_x_t(UMAX(plattice_x_t)))
                    continue;
                compared++;
                uint64_t const y = first_point(q, r, logI, M, si, sj);
                if (x != y && bad++ < 10)
                    fprintf(stderr, "m=%u q=%lu r=%lu logI=%d class (%u,%u):"
                            " starting_point %lu, expected %lu\n",
                            M, (unsigned long) q, (unsigned long) r, logI,
                            si, sj, (unsigned long) x, (unsigned long) y);
            }
        }
    }
    return bad;
}

int main(int argc, char const * argv[])
{
    cxx_param_list pl;
    pl.declare_usage("seed", "random seed (default 1)");
    pl.declare_usage("ntests", "number of random lattices (default 5000)");
    pl.process_command_line(argc, argv, false);
    unsigned long seed = 1;
    unsigned long ntests = 5000;
    pl.parse("seed", seed);
    pl.parse("ntests", ntests);
    if (pl.warn_unused())
        return EXIT_FAILURE;
    std::mt19937_64 rng(seed);
    unsigned long compared = 0;
    unsigned long bad = check<2>(rng, ntests, compared)
                      + check<3>(rng, ntests, compared)
                      + check<6>(rng, ntests / 4, compared);
    printf("%lu start points compared, %lu wrong\n", compared, bad);
    return bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
