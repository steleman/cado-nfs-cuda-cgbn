#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"
#include <gmp.h>

#include "cxx_mpz.hpp"
#include "ecm/facul.hpp"
#include "ecm/facul_strategies.hpp"
#include "macros.h"

/* The STOP marker of strategy files: "STOP" after a method
 * means that the pair is given up if that side has not split by then;
 * the methods after it only work on what is left after a split. And the
 * MPQS method, which splits whatever it is given.
 *
 * The cofactors are built so that P-1 behaves predictably: r - 1 is
 * smooth, so a small P-1 finds r; s is a safe prime with (s-1)/2 below
 * the B2 of a second P-1 but above the first one's; p is a safe prime
 * that no P-1 here finds. */

static cxx_mpz safe_prime(unsigned long start)
{
    cxx_mpz p, h;
    for (unsigned long q = start | 1;; q += 2) {
        mpz_set_ui(h, q);
        if (!mpz_probab_prime_p(h, 25))
            continue;
        mpz_mul_2exp(p, h, 1);
        mpz_add_ui(p, p, 1);
        if (mpz_probab_prime_p(p, 25))
            return p;
    }
}

static cxx_mpz smooth_prime()
{
    /* r - 1 = 2*3*5*7*11*13*17 * t with t a prime below 100: all of it
     * is in the exponent of P-1 stage 1 with B1 = 100 */
    cxx_mpz r, t;
    for (mpz_set_ui(t, 19); mpz_cmp_ui(t, 100) < 0; mpz_nextprime(t, t)) {
        mpz_mul_ui(r, t, 510510UL);
        mpz_add_ui(r, r, 1);
        if (mpz_probab_prime_p(r, 25))
            return r;
    }
    ASSERT_ALWAYS(0);
}

/* the status of side 1 of (1, N), with a strategy file whose only cell
 * is that pair of sizes */
static facul_status run(cxx_mpz const & N, std::string const & chain,
                        std::vector<cxx_mpz> & primes)
{
    std::vector<cxx_mpz> const n { cxx_mpz(1), N };
    std::string const text = fmt::format(
            "{} {}: {{ 1: {} }}\n",
            n[0].bits(), n[1].bits(), chain);

    // coverity[secure_temp]
    FILE * f = tmpfile();
    DIE_ERRNO_DIAG(f == nullptr, "tmpfile(%s)", "");
    fputs(text.c_str(), f);
    fflush(f);
    std::vector<unsigned long> const B(2, 1UL << 20);
    std::vector<unsigned int> const lpb(2, 32);
    std::vector<unsigned int> const mfb(2, 96);
    auto const F = facul_strategies(B, lpb, mfb, true, f, 0);
    fclose(f);

    auto res = facul_all(n, F);
    primes = res[1].primes;
    return res[1].status;
}

int main()
{
    int rc = EXIT_SUCCESS;
    auto check = [&rc](bool ok, std::string const & what) {
        if (!ok) {
            fmt::print(stderr, "error: {}\n", what);
            rc = EXIT_FAILURE;
        }
    };

    cxx_mpz const p = safe_prime(1UL << 29);    /* 31 bits */
    cxx_mpz const s = safe_prime(1UL << 14);    /* 16 bits */
    cxx_mpz const r = smooth_prime();
    std::vector<cxx_mpz> primes;

    std::string const first = "PM1 100 1000 ";
    std::string const stop = "STOP ";
    std::string const then = "PM1 1000 40000 ";

    /* r*p: the first P-1 splits, STOP lets the pair through */
    cxx_mpz N = r * p;
    check(run(N, first + stop + then, primes) == FACUL_SMOOTH
          && primes.size() == 2,
          fmt::format("{} = {} * {} should be smooth", N, r, p));

    /* s*p: without STOP, the second P-1 finds s */
    N = s * p;
    check(run(N, first + then, primes) == FACUL_SMOOTH && primes.size() == 2,
          fmt::format("{} = {} * {} should be smooth without STOP", N, s, p));

    /* with STOP, nothing splits by the marker, so the pair is given up */
    check(run(N, first + stop + then, primes) == FACUL_NOT_SMOOTH,
          fmt::format("{} = {} * {} should be given up at STOP", N, s, p));

    /* the methods after STOP do finish what a split leaves: r*s*p */
    N = r * s * p;
    check(run(N, first + stop + then, primes) == FACUL_SMOOTH
          && primes.size() == 3,
          fmt::format("{} = {} * {} * {} should be smooth", N, r, s, p));

    /* MPQS splits p*q, where no P-1 does; but not past a STOP */
    cxx_mpz const q = safe_prime(1UL << 28);    /* 30 bits */
    N = p * q;
    std::string const mpqs = "MPQS ";
    check(run(N, first + mpqs, primes) == FACUL_SMOOTH && primes.size() == 2,
          fmt::format("{} = {} * {} should be split by MPQS", N, p, q));
    check(run(N, first + stop + mpqs, primes) == FACUL_NOT_SMOOTH,
          fmt::format("{} = {} * {} should be given up at STOP", N, p, q));

    return rc;
}
