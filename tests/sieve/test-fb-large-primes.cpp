#include "cado.h" // IWYU pragma: keep

/* Check what bucket-sieving does with one factor base prime, for primes
 * near the top of the range that the factor base supports.
 *
 * For a prime p and a root r, the chain is: transform r to the q-lattice
 * (fb_entry_x_roots<>::transform_roots, or fb_root_in_qlattice for
 * projective roots), build and reduce the p-lattice (plattice_info), and
 * walk it with the Franke-Kleinjung enumerator (plattice_enumerator),
 * possibly within a sublattice. The positions visited are compared with
 * a direct computation of all (i,j) in the sieve region such that
 *      a == r*b mod p        (affine root)
 *      a*r == b mod p        (projective root)
 * where (a,b) = i*(a0,b0) + j*(a1,b1). Since p is larger than I, there
 * are only few solutions per row, so this is cheap.
 *
 * Primes are drawn in [pmin, pmax], and the bounds themselves are always
 * tested, as well as roots that give a projective or a vertical p-lattice.
 */

#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <gmp.h>
#include "fmt/base.h"
#include "fmt/format.h"

#include "cado_main.hpp"
#include "cxx_mpz.hpp"
#include "fb-types.hpp"
#include "gmp_aux.h"
#include "las-arith.hpp"
#include "las-fbroot-qlattice.hpp"
#include "las-plattice.hpp"
#include "las-qlattice.hpp"
#include "las-sublat.hpp"
#include "macros.h"
#include "params.hpp"
#include "utils_cxx.hpp"

namespace {

struct test_setup {
    int logI = 16;
    uint32_t J = 1U << 15;
    bool verbose = false;
    cxx_gmp_randstate rstate;
};

uint64_t random_below(cxx_gmp_randstate & rstate, uint64_t n)
{
    cxx_mpz t = n;
    mpz_urandomm(t, rstate, t);
    return uint64_t(t);
}

/* -1/p mod 2^32, the redc constant for the factor base entries */
redc_invp_t make_invp(fbprime_t p)
{
    cxx_mpz const m = cxx_mpz(1) << (8 * sizeof(redc_invp_t));
    return static_cast<redc_invp_t>(uint64_t(m - cxx_mpz(p).invmod(m)));
}

/* A random basis with 31-bit entries, whose determinant is coprime to
 * p, as fb_root_in_qlattice_31bits wants. */
qlattice_basis random_basis(cxx_gmp_randstate & rstate, fbprime_t p)
{
    for (;;) {
        auto c = [&]() {
            return int64_t(random_below(rstate, UINT64_C(1) << 32)) -
                   (INT64_C(1) << 31);
        };
        qlattice_basis B(c(), c(), c(), c());
        cxx_mpz const det = cxx_mpz(B.a0) * B.b1 - cxx_mpz(B.a1) * B.b0;
        if (det % p == 0)
            continue;
        return B;
    }
}

/* Reduce x modulo p into [0,p) (x % p would truncate) */
uint64_t mod_p(cxx_mpz const & x, fbprime_t p)
{
    return mpz_fdiv_uint64(x, p);
}

/* For each row j, the hits on the row are the i such that
 * i * alpha + j * beta == 0 mod p. Return (alpha, beta) mod p.
 */
std::array<uint64_t, 2> row_congruence(fbprime_t p, fb_root_p1 R,
                                       qlattice_basis const & B)
{
    cxx_mpz const r = R.r;
    /* affine: a - r*b ; projective: a*r - b */
    if (R.is_affine())
        return { mod_p(cxx_mpz(B.a0) - r * B.b0, p),
                 mod_p(cxx_mpz(B.a1) - r * B.b1, p) };
    else
        return { mod_p(r * B.a0 - B.b0, p), mod_p(r * B.a1 - B.b1, p) };
}

/* The expected positions, as x = (i' + I/2) + (j' << logI) where
 * (i, j) = (m*i' + si0, m*j' + sj0). Without sublattices, the point
 * (0,0) and the half-row j=0, i<0 are not part of the sieve region.
 */
std::vector<plattice_x_t> expected_hits(test_setup const & S, fbprime_t p,
                                        fb_root_p1 R, qlattice_basis const & B,
                                        sublat_runtime_t const & sublat)
{
    auto const [alpha, beta] = row_congruence(p, R, B);
    int64_t const I = int64_t(1) << S.logI;
    int64_t const m = sublat.m;
    /* real i ranges over [imin, imin + m*I) */
    int64_t const imin = -m * (I / 2) + sublat.i0;
    std::vector<plattice_x_t> res;

    auto push = [&](uint64_t jj, int64_t i) {
        int64_t const ii = (i - int64_t(sublat.i0)) / m;
        if (m == 1 && jj == 0 && ii <= 0)
            return;
        res.push_back(plattice_x_t(ii + I / 2) + (plattice_x_t(jj) << S.logI));
    };

    if (alpha == 0) {
        /* hits are on the rows j == 0 mod p, for all i. Since p > J,
         * that is only j == 0. */
        if (sublat.j0 == 0)
            for (int64_t i = imin; i < imin + m * I; i += m)
                push(0, i);
        return res;
    }

    /* On row j, i == c_j := -j*beta/alpha mod p. With j = m*jj + sj0,
     * c_j = c0 + jj * d mod p. */
    cxx_mpz const t = cxx_mpz(alpha).invmod(p) * (p - beta);
    uint64_t c = mod_p(t * sublat.j0, p);
    uint64_t const d = mod_p(t * m, p);
    uint64_t const imin_mod_p = mod_p(cxx_mpz(imin), p);

    for (uint64_t jj = 0; jj < S.J; jj++) {
        /* smallest i >= imin with i == c mod p, then also i == si0 mod m */
        int64_t i = imin + int64_t(c >= imin_mod_p ? c - imin_mod_p
                                                   : c + p - imin_mod_p);
        for (; i < imin + m * I; i += int64_t(p))
            if (((i - int64_t(sublat.i0)) % m) == 0)
                push(jj, i);
        c += d;
        if (c >= p)
            c -= p;
    }
    std::ranges::sort(res);
    return res;
}

/* The positions that the production code visits for this lattice.
 * This mirrors the logic in fill_in_buckets_toplevel_impl, except that
 * the special cases are walked with the generic enumerator. Returns
 * false if the production code would not sieve this lattice at all
 * (degenerate lattices under sublattices).
 */
template <uint32_t M>
bool visited_hits(test_setup const & S, fbprime_t q, fb_root_p1 R,
                  sublat_t<M> const & sublat, std::vector<plattice_x_t> & res)
{
    res.clear();
    plattice_info const pli(q, R.r, R.proj, S.logI);
    plattice_enumerator::fence const F(S.logI, S.J);
    plattice_enumerator ple(pli, 0, S.logI, sublat);
    if constexpr (M > 1) {
        if (ple.is_projective_like(S.logI) || pli.is_vertical_line(S.logI))
            return false;
        /* the dense storage that replays this lattice for the classes
         * other than the first one must give it back unchanged */
        plattice_info_dense_t const D(pli, 0, q);
        plattice_info const pli2 = D.unpack(q);
        if (plattice_info(pli2) != pli)
            throw cado::error("dense storage of the p-lattice is lossy");
    } else {
        ple.next(F);
    }
    if (pli.is_discarded())
        return true;
    for (; !ple.done(F); ple.next(F))
        res.push_back(ple.get_x());
    return true;
}

/* Transform the roots of one prime to the q-lattice, the same way as
 * fb_entry_x_roots<>::transform_roots does (which we can't call here,
 * since it lives in the las libraries): first a batch transform, and
 * the roots one at a time if that fails or if a root is projective.
 * Check against the reference implementation. */
template <int N>
std::array<fb_root_p1, N>
transformed_roots(fbprime_t p, std::array<fb_root_p1, N> const & roots,
                  qlattice_basis const & B)
{
    std::array<fb_root_p1, N> res;
    redc_invp_t const invp = make_invp(p);
    bool const all_affine = std::ranges::all_of(
        roots, [](fb_root_p1 const & R) { return R.is_affine(); });
    std::array<fbroot_t, N> rr, rt;
    for (int i = 0; i < N; i++)
        rr[i] = roots[i].r;
    if (all_affine &&
        fb_root_in_qlattice_batch(rt.data(), p, rr.data(), invp, B, N)) {
        for (int i = 0; i < N; i++)
            res[i] = fb_root_p1(rt[i]);
    } else {
        for (int i = 0; i < N; i++)
            res[i] = fb_root_in_qlattice(p, roots[i], invp, B);
    }
    for (int i = 0; i < N; i++) {
        auto const ref = reference_fb_root_in_qlattice(p, roots[i], B);
        if (ref.r != res[i].r || ref.proj != res[i].proj)
            throw cado::error("root transform of ({}{}) mod {} gives ({}{}),"
                              " expected ({}{})",
                              roots[i].r, roots[i].proj ? ",proj" : "", p,
                              res[i].r, res[i].proj ? ",proj" : "", ref.r,
                              ref.proj ? ",proj" : "");
    }
    return res;
}

struct counters {
    unsigned long roots = 0;
    unsigned long hits = 0;
    unsigned long skipped = 0;
};

template <int N>
void one_test(test_setup & S, fbprime_t p, std::array<fb_root_p1, N> roots,
              qlattice_basis const & B, counters & C)
{
    auto const T = transformed_roots<N>(p, roots, B);
    for (int k = 0; k < N; k++) {
        C.roots++;
        for (sublat_runtime_t const & sl: { sublat_runtime_t(1, 0, 0),
                                             sublat_runtime_t(2, 1, 0),
                                             sublat_runtime_t(3, 2, 1),
                                             sublat_runtime_t(6, 5, 4) }) {
            std::vector<plattice_x_t> got;
            bool const sieved = dispatch_sublat(sl, [&](auto sublat) {
                return visited_hits(S, p, T[k], sublat, got);
            });
            if (!sieved) {
                C.skipped++;
                continue;
            }
            auto const want = expected_hits(S, p, roots[k], B, sl);
            C.hits += want.size();
            if (got != want) {
                std::string s = fmt::format(
                    "p={} root=({}{}) transformed=({}{}) basis={} {}:"
                    " {} positions visited, {} expected",
                    p, roots[k].r, roots[k].proj ? ",proj" : "", T[k].r,
                    T[k].proj ? ",proj" : "", B, sl, got.size(), want.size());
                std::vector<plattice_x_t> d;
                std::ranges::set_symmetric_difference(got, want,
                                                      std::back_inserter(d));
                for (size_t i = 0; i < std::min(d.size(), size_t(8)); i++)
                    s += fmt::format("\n    x={} (i={}, j={}) {}", d[i],
                                     int64_t(d[i] & ((1U << S.logI) - 1)) -
                                         (int64_t(1) << (S.logI - 1)),
                                     d[i] >> S.logI,
                                     std::ranges::binary_search(want, d[i])
                                         ? "missed"
                                         : "spurious");
                throw cado::error(s);
            }
        }
    }
}

fbprime_t random_prime(cxx_gmp_randstate & rstate, uint64_t pmin,
                       uint64_t pmax)
{
    for (;;) {
        cxx_mpz t = pmin + random_below(rstate, pmax - pmin + 1);
        mpz_nextprime(t, t);
        if (t <= pmax)
            return static_cast<fbprime_t>(uint64_t(t));
    }
}

/* largest prime <= x, or 0 */
fbprime_t prev_prime(uint64_t x)
{
    for (cxx_mpz t = x; t >= 2; t -= 1)
        if (mpz_probab_prime_p(t, 25))
            return static_cast<fbprime_t>(uint64_t(t));
    return 0;
}

/* r such that a0 - r*b0 == 0 mod p (projective transformed root), or
 * a1 - r*b1 == 0 mod p (transformed root zero, i.e. a vertical
 * p-lattice) */
fbroot_t special_root(fbprime_t p, qlattice_basis const & B, bool vertical)
{
    cxx_mpz den = vertical ? B.b1 : B.b0;
    if (!mpz_invert(den, den, cxx_mpz(p)))
        return 0;
    return static_cast<fbroot_t>(mod_p(den * (vertical ? B.a1 : B.a0), p));
}

void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage("pmin", "test primes >= pmin (default 2^31-2^20)");
    pl.declare_usage("pmax", "test primes <= pmax (default 2^32-1)");
    pl.declare_usage("ntests", "number of random primes (default 200)");
    pl.declare_usage("I", "set logI (default 16)");
    pl.declare_usage("J", "set J (default 2^15)");
    pl.declare_usage("seed", "random seed (default 1)");
    pl.declare_usage("v", "verbose mode");
}

} // namespace

static int main_(int argc, char const * argv[]);

int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;
    declare_usage(pl);
    pl.configure_switch("-v");
    pl.process_command_line(argc, argv, false);

    test_setup S;
    uint64_t pmin = (UINT64_C(1) << 31) - (UINT64_C(1) << 20);
    uint64_t pmax = FBPRIME_MAX;
    unsigned long ntests = 200;
    unsigned long seed = 1;
    pl.parse("pmin", pmin);
    pl.parse("pmax", pmax);
    pl.parse("ntests", ntests);
    pl.parse("I", S.logI);
    pl.parse("J", S.J);
    pl.parse("seed", seed);
    pl.parse("-v", S.verbose);
    if (pl.warn_unused())
        return EXIT_FAILURE;

    ASSERT_ALWAYS(pmax <= FBPRIME_MAX);
    /* the reference enumeration assumes at most a few hits per row */
    ASSERT_ALWAYS(pmin > (UINT64_C(6) << S.logI));
    ASSERT_ALWAYS(pmin <= pmax);

    gmp_randseed_ui(S.rstate, seed);

    std::vector<fbprime_t> primes;
    for (fbprime_t const p: { prev_prime(pmax), prev_prime(pmin + 1000) })
        if (p >= pmin)
            primes.push_back(p);
    /* also the primes around the threshold where redc_32 needs a carry
     * check. */
    for (fbprime_t const p:
         { prev_prime(UINT64_C(1) << 31), prev_prime((UINT64_C(1) << 31) + 100) })
        if (p >= pmin && p <= pmax)
            primes.push_back(p);
    for (unsigned long k = 0; k < ntests; k++)
        primes.push_back(random_prime(S.rstate, pmin, pmax));

    counters C;
    for (fbprime_t const p: primes) {
        qlattice_basis const B = random_basis(S.rstate, p);
        auto rnd = [&]() { return fbroot_t(random_below(S.rstate, p)); };
        /* simple entries, with one and two roots */
        one_test<1>(S, p, { fb_root_p1(rnd()) }, B, C);
        one_test<2>(S, p, { fb_root_p1(rnd()), fb_root_p1(rnd()) }, B, C);
        /* a projective root, and roots that give a projective or a
         * vertical p-lattice */
        one_test<3>(S, p,
                    { fb_root_p1(rnd(), true),
                      fb_root_p1(special_root(p, B, false)),
                      fb_root_p1(special_root(p, B, true)) },
                    B, C);
    }

    fmt::print("{} primes in [{}, {}], {} roots, {} hits checked,"
               " {} degenerate lattices skipped under sublattices\n",
               primes.size(), pmin, pmax, C.roots, C.hits, C.skipped);

    return EXIT_SUCCESS;
}
