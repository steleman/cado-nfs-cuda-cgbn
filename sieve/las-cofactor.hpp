#ifndef CADO_LAS_COFACTOR_HPP
#define CADO_LAS_COFACTOR_HPP

// IWYU pragma: no_include <ext/alloc_traits.h>

#include <cstdio>
#include <cstdint>

#include <mutex>
#include <vector>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "ecm/facul.hpp"
#include "las-siever-config.hpp"
#include "params.hpp"
#include "ecm/facul_strategies.hpp"

class cofactorization_statistics {
    /* We rarely use this, if ever. Do we want to generalize this at all?
     * How? For the moment we just bail out if more than two sides.
     */
    FILE * file;
    FILE * dumpfile = nullptr;
    std::vector<std::vector<uint32_t>> cof_call;
    std::vector<std::vector<uint32_t>> cof_success;
    std::mutex lock;
public:
    cofactorization_statistics(cxx_param_list & pl);
    bool active() { return file != nullptr; }
    void call(int bits0, int bits1);
    /* write one line "a b N0 N1 ..." per pair that enters cofactoring,
     * for replaying the cofactorization step offline */
    void dump(int64_t a, uint64_t b, std::vector<cxx_mpz> const & norm);
    void print();
    void call(std::vector<cxx_mpz> const & norm, std::vector<int> & cof_bitsize) {
        if (!active()) return;
        ASSERT_ALWAYS(norm.size() == 1 || norm.size() == 2);
        cof_bitsize.assign(norm.size(), 0);
        cof_bitsize[0] = mpz_sizeinbase(norm[0], 2);
        if (norm.size() == 2) {
            cof_bitsize[1] = mpz_sizeinbase(norm[1], 2);
        } else {
            cof_bitsize.push_back(0);
        }
        call(cof_bitsize[0], cof_bitsize[1]);
    }
    void success(std::vector<int> const & cof_bitsize) {
        if (!active()) return;
        ASSERT_ALWAYS(cof_bitsize.size() == 2);
        cof_success[cof_bitsize[0]][cof_bitsize[1]]++;
    }
    void success(int bits0, int bits1)
    {
        if (!active()) return;
        cof_success[bits0][bits1]++;
    }
    ~cofactorization_statistics();
    static void declare_usage(cxx_param_list & pl);
};

int check_leftover_norm (cxx_mpz const & n, siever_side_config const & sc);

facul_status factor_leftover_norms(
        std::vector<cxx_mpz> const & norms,
        std::vector<std::vector<cxx_mpz>> &,
        std::vector<unsigned long> const &,
        facul_strategies const &);

/* handy shortcut. Can't have it defined at the facul.hpp level because
 * facul does not know about las stuff. */
inline facul_strategies * facul_make_strategies (siever_config const & conf, FILE* file, int verbose);
inline facul_strategies * facul_make_strategies (siever_config const & conf, FILE* file, int verbose)
{
    std::vector<unsigned long> lim(conf.sides.size());
    std::vector<unsigned int> lpb(conf.sides.size());
    std::vector<unsigned int> mfb(conf.sides.size());
    std::vector<int> ncurves(conf.sides.size());
    auto plim = lim.begin();
    auto plpb = lpb.begin();
    auto pmfb = mfb.begin();
    auto pncurves = ncurves.begin();
    for(auto const & s : conf.sides) {
        *plim++ = s.lim;
        *plpb++ = s.lpb;
        *pmfb++ = s.mfb;
        *pncurves++ = s.ncurves;
    }

    /* "Perfectly sieved" would let facul take any cofactor below lim^2
     * for a prime without testing it. las cannot promise that: besides
     * the primes that sublattices skip, the special-q prime itself is
     * never sieved nor divided out again (only the factors of a
     * composite q are, in cofactoring_sync), so with q < lim the leftover
     * can be q times a prime above lim, a composite below lim^2. This
     * used to read (conf.sublat_bound == 0), which is never true (no
     * sublattices is sublat_bound == 1), so the value was always false;
     * it is now false on purpose. */
    bool const perfectly_sieved = false;
    if (file) {
        return new facul_strategies(
                lim,
                lpb,
                mfb,
                perfectly_sieved,
                file, verbose);
    } else {
        return new facul_strategies(
                lim,
                lpb,
                mfb,
                ncurves,
                perfectly_sieved,
                verbose);
    }
}

#endif
