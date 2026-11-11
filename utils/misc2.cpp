#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstddef>

#include <vector>
#include <algorithm>
#include <utility>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "getprime.h"
#include "misc.h"
#include "runtime_numeric_cast.hpp"
#include "random_distributions.hpp"

/* The logarithmic integral li(p) = Ei(log p), which approximates the
 * number of primes below p.
 *
 * We compute it here rather than call std::expint. The C++ mathematical
 * special functions are an optional part of the standard library and
 * libc++ does not provide them, so on a platform without them this used
 * to fall back on a four-term asymptotic expansion -- a different
 * function, low by about 1% for p in the thousands. Everything that
 * sizes a structure from this estimate, and everything sieve/strategies
 * prints, then came out differently there.
 *
 * The series below converges for every p > 1, its terms are all
 * positive so nothing cancels, and it reaches the precision of a double
 * in a few dozen terms. It needs exp(log p) = p worth of range
 * internally, so it holds for any p a double can represent at all.
 */
static double logarithmic_integral(double p)
{
    if (p <= 1)
        return 0;

    constexpr double euler_gamma = 0.57721566490153286061;

    /* Ei(x) = gamma + log(x) + sum_{k >= 1} x^k / (k * k!) */
    const double x = log(p);
    double sum = 0;
    double term = 1;
    for (int k = 1; k < 1000; k++) {
        term *= x / k;
        const double contribution = term / k;
        sum += contribution;
        if (contribution <= sum * 1.0e-17)
            break;
    }

    return euler_gamma + log(x) + sum;
}

double nprimes_interval(double p0, double p1)
{
    return logarithmic_integral(p1) - logarithmic_integral(p0);
}

/* returns the number of primes <= 2^n ; result is exact up to some
 * bound. Note that exact counts are known for larger values as well (see
 * test_prime_count.cpp). Anyway the output that we get is accurate to a
 * relative precision of 2^-20.
 */
double prime_pi_2exp(unsigned int n)
{
    static const double A7053[] = {
        0, 1, 2, 4, 6, 11, 18, 31,
        54, 97, 172, 309, 564, 1028, 1900, 3512,
        6542, 12251, 23000, 43390, 82025, 155611, 295947, 564163,
        1077871, 2063689, 3957809, 7603553,
        14630843, 28192750, 54400028, 105097565,
        203280221, 393615806, 762939111, 1480206279,
        2874398515, 5586502348, 10866266172, 21151907950,
        41203088796, 80316571436, 156661034233, 305761713237,
        /*
        597116381732, 1166746786182, 2280998753949, 4461632979717,
        8731188863470, 17094432576778, 33483379603407, 65612899915304,
        128625503610475,
        */
        };
    /* there's more at https://oeis.org/A007053/b007053.txt */
    if (n < sizeof(A7053) / sizeof(A7053[0]))
        return A7053[n];
    else
        return nprimes_interval(2, ldexp(1, runtime_numeric_cast<int>(n)));
}

/* generate a random i-bit integer that roughly follows the distribution
 * of i-bit prime numbers
 */
double random_along_prime_distribution(unsigned int bits, gmp_randstate_t rstate)
{
    const double n0 = prime_pi_2exp(bits - 1);
    const double n1 = prime_pi_2exp(bits);
    double a, b, n;

    /* we assume the n-th prime is in a*n*log(n)+b, thus we want:
     *    a*n0*log(n0) + b = 2^(bits-1)
     *    a*n1*log(n1) + b = 2^bits
     */
    a = ldexp(1.0, runtime_numeric_cast<int>(bits) - 1) / (n1 * log(n1) - n0 * log(n0));
    b = ldexp(1.0, runtime_numeric_cast<int>(bits)) - a * n1 * log(n1);
    n = n0 + (n1 - n0) * random_uniform(rstate);
    return a * n * log(n) + b;
}

std::vector<unsigned long> subdivide_primes_interval(unsigned long p0, unsigned long p1, size_t n)
{
    std::vector<unsigned long> ret;
    ret.push_back(p0);
    unsigned long const previous = p0;
    double const total_count = nprimes_interval(
            (double) (p0),
            (double) (p1));
    /* by proceeding like this, we're wasting time, since p1 always
     * serves as an endpoint, so that we have a complexity which is
     * roughly n * log(p1-p0). We could have something like log(p1-p0) +
     * 2*log((p1-p0)/2) + 4 * log((p1-p0)/4) + ... which would save a
     * lower-order additive term. No big deal, really.
     */
    for(size_t i = 1 ; i < n ; i++) {
        /* find smallest p such that nprimes_interval(previous, p) >= i *
         * total_count / n ; do simple dichotomy.
         */
        double const target = i * total_count / n;
        unsigned long q0 = previous;
        unsigned long q1 = p1;
        unsigned long q = previous + (p1 - previous) / (n - i);
        for( ; q > q0 ; ) {
            double const r = nprimes_interval(
                    (double) (p0),
                    (double) (q));
            if (r < target)
                q0 = q;
            else
                q1 = q;
            q = (q0 + q1) / 2;
        }
        ret.push_back(q);
    }
    ret.push_back(p1);
    return ret;
}

/* This is meant to be used by C code */
void subdivide_primes_interval_proxy(unsigned long * r, unsigned long p0, unsigned long p1, size_t n)
{
    auto v = subdivide_primes_interval(p0, p1, n);
    std::ranges::copy(v, r);
}



std::vector<std::pair<cxx_mpz, int> > trial_division(cxx_mpz const& n0, unsigned long B, cxx_mpz & cofactor)/*{{{*/
{
    std::vector<std::pair<cxx_mpz, int> > res;
    prime_info pinf;

    prime_info_init (pinf);
    cxx_mpz n = n0;

    /* if n takes k bits it means that n < 2^k. If p^2>=2^k, then we
     * can break. A sufficient condition for this is
     * p^2 >= 2^(2*(ceiling(k/2)))
     * p   >= 2^(  (floor((k+1)/2)))
     * because regardless of the parity of k, this implies our
     * condition
     */
    unsigned long bound_shift = (mpz_sizeinbase(n, 2) + 1) / 2;
    for (unsigned long p = 2; p < B; p = getprime_mt (pinf)) {
        if (bound_shift < ULONG_BITS && p >> bound_shift) {
            if (mpz_cmpabs_ui(n, 1U) > 0 && mpz_cmpabs_ui(n, B) < 0) {
                res.emplace_back(n, 1);
                n = 1U;
                if (mpz_sgn(n0) < 0) {
                    mpz_neg(res.back().first, res.back().first);
                    mpz_neg(n, n);
                }
            }
            break;
        }
        if (!mpz_divisible_ui_p(n, p)) continue;
        int k = 0;
        for( ; mpz_divisible_ui_p(n, p) ; mpz_fdiv_q_ui(n, n, p), k++);
        bound_shift = (mpz_sizeinbase(n, 2) + 1) / 2;
        res.emplace_back(p, k);
    }
    cofactor = n;

    prime_info_clear (pinf); /* free the tables */
    return res;
}
/*}}}*/
