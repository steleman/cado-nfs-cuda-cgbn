#include "cado.h" // IWYU pragma: keep

#ifndef SUPPORT_LARGE_Q
#include <cinttypes>
#endif
#include <cstddef>

#include <array>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include "fmt/format.h"

#include <gmp.h>

#include "ecm/batch.hpp"
#ifndef SUPPORT_LARGE_Q
#include "gcd.h"
#endif
#ifdef SUPPORT_LARGE_Q
#include "cxx_mpz.hpp"
#endif
#include "las-cofac-standalone.hpp"
#include "las-cofactor.hpp"
#include "las-coordinates.hpp"
#include "las-siever-config.hpp"
#include "las-threads-work-data.hpp"
#include "special-q.hpp"
#include "las-where-am-i-proxy.hpp"
#include "lock_guarded_container.hpp"
#include "relation.hpp"
#include "macros.h"

struct qlattice_basis; // IWYU pragma: keep

/* This is one input to the late cofactoring process (aka ECM). Here, we
 * mean the stuff that is done detached from the rest of the siever
 * stuff: we no longer care about purging buckets and so on, these may
 * safely be used for later work.
 */
cofac_standalone::cofac_standalone() : a(0), b(0) {/*{{{*/
#ifdef SUPPORT_LARGE_Q
    mpz_set_ui(az, 0);
    mpz_set_ui(bz, 0);
#endif
}/*}}}*/
cofac_standalone::cofac_standalone(int nsides, int N, size_t x, int logI,
                                   special_q_data_base const & Q)
    : S(nsides, 0)
    , norm(nsides, 0)
    , factors(nsides)
    , lps(nsides)
{/*{{{*/
    convert_Nx_to_ab (a, b, N, x, logI, Q);
#ifdef SUPPORT_LARGE_Q
    convert_Nx_to_abmpz (az, bz, N, x, logI, Q);
#endif
}/*}}}*/
bool cofac_standalone::trace_on_spot() const {/*{{{*/
#ifdef SUPPORT_LARGE_Q
    return extern_trace_on_spot_ab(az, bz);
#else
    return extern_trace_on_spot_ab(a, b);
#endif
}/*}}}*/
bool cofac_standalone::gcd_coprime_with_q(special_q const & E) const {/*{{{*/
    /* Since the q-lattice is exactly those (a, b) with
       a == rho*b (mod q), q|b  ==>  q|a  ==>  q | gcd(a,b) */
    /* In case of composite sq, have to check all factors... */
    /* FIXME: fast divisibility test here! */
    if (E.is_prime()) {
#ifndef SUPPORT_LARGE_Q
        if (b == 0 || (mpz_cmp_ui(E.p, b) <= 0 && b % mpz_get_ui(E.p) == 0))
#else
            if ((mpz_cmp_ui(bz, 0) == 0) || 
                    (mpz_cmp(E.p, bz) <= 0 &&
                     mpz_divisible_p(bz, E.p)))
#endif
                return false;
    } else {
#ifdef SUPPORT_LARGE_Q
        if (mpz_cmp_ui(bz, 0) == 0)
            return false;
        for (auto const& facq : E.prime_factors) {
            if ((mpz_cmp_ui(bz, facq) >= 0) && (mpz_divisible_ui_p(bz, facq))) {
                return false;
            }
        }
#else
        if (b == 0)
            return false;
        for (auto const& facq : E.prime_factors) {
            if (facq <= b && b % facq == 0) {
                return false;
            }
        }
#endif
    }
    return true;
}/*}}}*/
bool cofac_standalone::ab_coprime() const {/*{{{*/
#ifndef SUPPORT_LARGE_Q
    return bin_gcd_int64_safe(a,b) == 1;
#else
    cxx_mpz g;
    mpz_gcd(g, az, bz);
    return mpz_cmp_ui(g, 1) == 0;
#endif
}/*}}}*/
void cofac_standalone::print_as_survivor(FILE * f) {/*{{{*/
#ifndef SUPPORT_LARGE_Q
    gmp_fprintf(f, "%" PRId64 " %" PRIu64, a, b);
#else
    gmp_fprintf(f, "%Zd %Zd", (mpz_srcptr) az, (mpz_srcptr) bz);
#endif
    for (auto const & n: norm)
        gmp_fprintf(f, " %Zd", (mpz_srcptr) n);
    fprintf(f, "\n");
}/*}}}*/
relation cofac_standalone::get_relation(special_q const & doing) const {/*{{{*/
#ifndef SUPPORT_LARGE_Q
    relation rel(a, b);
#else
    relation rel(az, bz);
#endif

    /* Note that we explicitly do not bother about storing r in
     * the relations below */
    for (unsigned int side = 0; side < std::min(rel.sides.size(), factors.size()); side++) { // FIXME workaround for HARDCODED 2
        for (auto const& z : factors[side])
            rel.add(side, z, 0);
        for (auto const& z : lps[side])
            rel.add(side, z, cxx_mpz(0));
    }
    if (doing.is_prime()) {
        rel.add(doing.side, doing.p, cxx_mpz(0));
    } else {
        for (auto const& facq : doing.prime_factors)
            rel.add(doing.side, facq, cxx_mpz(0));
    }

    rel.compress();
    return rel;
}/*}}}*/
bool cofac_standalone::format_relation(special_q const & doing MAYBE_UNUSED, std::string & out MAYBE_UNUSED) const {/*{{{*/
#ifdef SUPPORT_LARGE_Q
    return false;
#else
    if (factors.size() != 2 || lps.size() != 2)
        return false;
    if (doing.side < 0 || doing.side > 1)
        return false;
    /* the primes of each side; relation::compress sorts them (all with
     * r = 0) and merges equal ones into exponents, which operator<<
     * prints again as repeats: so this is the sorted multiset */
    std::array<std::vector<uint64_t>, 2> P;
    for (int side = 0; side < 2; side++) {
        auto & v = P[side];
        v.reserve(factors[side].size() + lps[side].size() + 2);
        v.insert(v.end(), factors[side].begin(), factors[side].end());
        for (auto const & z : lps[side]) {
            if (!mpz_fits_ulong_p(z))
                return false;
            v.push_back(mpz_get_ui(z));
        }
    }
    if (doing.is_prime()) {
        if (!mpz_fits_ulong_p(doing.p))
            return false;
        P[doing.side].push_back(mpz_get_ui(doing.p));
    } else {
        for (auto const q : doing.prime_factors)
            P[doing.side].push_back(q);
    }
    auto it = std::back_inserter(out);
    fmt::format_to(it, "{},{}", a, b);
    for (auto & v : P) {
        std::ranges::sort(v);
        out += ':';
        bool comma = false;
        for (auto const p : v) {
            if (comma) out += ',';
            fmt::format_to(it, "{:x}", p);
            comma = true;
        }
    }
    return true;
#endif
}/*}}}*/
void cofac_standalone::transfer_to_cofac_list(lock_guarded_container<std::list<cofac_candidate>> & L) {/*{{{*/
    std::lock_guard<std::mutex> const foo(L.mutex());
    L.emplace_back(a, b, norm);
#if 0
    /* make sure threads don't write the cofactor list at the
     * same time !!! */
#warning "possible contention"
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&lock);
    cofac_list_add (L, a, b, norm, doing.side, doing.p);
    pthread_mutex_unlock(&lock);
#endif
}/*}}}*/
int cofac_standalone::factor_leftover_norms(nfs_work_cofac & wc) {/*{{{*/
    /* This proxies to las-cofactor.cpp */
    std::vector<unsigned long> Bs;
    Bs.reserve(wc.sc.sides.size());
    for (auto const & s: wc.sc.sides)
        Bs.push_back(s.lim);
    return ::factor_leftover_norms(norm,
            lps,
            Bs,
            *wc.strategies);
}/*}}}*/

std::ostream& operator<<(std::ostream& os, cofac_standalone const & cur)
{
#ifdef SUPPORT_LARGE_Q
    return os << "(" << cur.az << ", " << cur.bz << ")";
#else
    return os << "(" << cur.a << ", " << cur.b << ")";
#endif
}
