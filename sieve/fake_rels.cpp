#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>
#include <cstring> /* for strcmp() */
#include <cmath> /* for sqrt and floor and log and ceil */
#include <cstdint>

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmp.h>
#include "fmt/base.h"

#include "cado_poly.hpp"
#include "cxx_mpz.hpp"
#include "gmp_aux.h"
#include "fstream_maybe_compressed.hpp"
#include "indexed_relation.hpp"
#include "special-q.hpp"
#include "macros.h"
#include "misc.h"
#include "params.hpp"
#include "relation.hpp"
#include "las-special-markers.hpp"
#include "renumber.hpp"
#include "timing.h"
#include "typedefs.h"
#include "verbose.hpp"
#include "utils_cxx.hpp"
#include "cado_main.hpp"

static int verbose = 0; /* verbosity level */

/*
 * The goal of this binary is to produce relations that try to be good
 * approximations of what las would produce, with respect to filtering.
 * The idea is to feed the purge/merge steps of the filtering with these
 * relations to get an idea of the final matrix that will come out of the
 * sieving step for a given set of parameters.
 *
 * The shell script cado-nfs/scripts/estimate_matsize/estimate_matsize.sh
 * is an attempt to run all the required steps for such a simulation:
 * sampling with las, fake relation generation with this binary, and
 * filter.
 *
 * The binary takes as input:
 *   - poly file
 *   - lpb on each side
 *   - a range of special-q for a given side
 *   - a sample of relations (output of las) for this range
 *   - the renumber table.
 * The renumber table is required, because this binary will produce
 * relations as if they were coming out of dup2 (hence renumbered).
 *
 * The sample of relations must also be de-duplicated, therefore the -dup
 * option of las must be used for the sampling.
 *
 */

// A global mutex for I/O
static std::mutex io_mutex;

/* With -virtual, there is no renumber table: each prime below the large
 * prime bound of a side has one ideal on that side, which is what the
 * table has on average, and the ideals are in the order of the table, by
 * prime and then by side. With N(p) the number of primes below p (exact
 * below 2^24, li(p) above), and K[s] = N(2^lpb[s]), the k-th prime has
 * index_of(k, s) on side s: below both bounds, the sides alternate, and
 * above the smaller bound only one side is left. perturb() replaces the
 * ideals of a sample relation by random ideals of about the same size,
 * so only the density of the ideals matters, and that is right. Two
 * ideals of the same norm on the same side get the same index: with a
 * shrink factor, such ideals are merged anyway. This makes it possible
 * to simulate parameters whose renumber table would not fit */
struct virtual_numbering {
    static constexpr uint64_t small_bound = UINT64_C(1) << 24;
    std::vector<uint32_t> small_primes;
    long double li_small = 0;
    std::array<uint64_t, 2> K {};

    static long double li(long double x) {
        /* li(x) = gamma + log(log(x)) + sum_{k>=1} log(x)^k / (k k!) */
        long double const L = std::log(x);
        long double term = 1, sum = 0;
        for (int k = 1 ; k < 400 ; k++) {
            term *= L / k;
            long double const t = term / k;
            sum += t;
            if (t < sum * 1e-21L)
                break;
        }
        return 0.57721566490153286060651209L + std::log(L) + sum;
    }

    explicit virtual_numbering(std::array<int, 2> const & lpb) {
        std::vector<bool> composite(small_bound, false);
        for (uint64_t p = 2 ; p < small_bound ; p++) {
            if (composite[p])
                continue;
            small_primes.push_back(uint32_t(p));
            for (uint64_t m = p * p ; m < small_bound ; m += p)
                composite[m] = true;
        }
        li_small = li((long double) small_bound);
        for (int side = 0 ; side < 2 ; side++)
            K[side] = N(UINT64_C(1) << lpb[side]);
    }

    /* the number of primes below p (p need not be prime) */
    uint64_t N(uint64_t p) const {
        if (p < small_bound)
            return std::ranges::lower_bound(small_primes, uint32_t(p)) - small_primes.begin();
        return small_primes.size() + uint64_t(std::llround(li((long double) p) - li_small));
    }

    /* the k-th prime (from 0) on side s, which must be below K[s] */
    index_t index_of(uint64_t k, int side) const {
        return std::min(k, K[0]) + std::min(k, K[1]) + (side == 1 && k < K[0]);
    }

    index_t index(uint64_t p, int side) const {
        return index_of(N(p), side);
    }
};

// A global variable for the number of relations printed
static unsigned long rels_printed = 0;

// Structure that contains indices (in the sense of renumber.[ch]) for
// one side, with the possibility to get a randomized ideal on the same
// side, and more or less of the same size.
// Usage:
//  - init()
//  - do many append() (increasing order of indices) and append_prime()
//  - finalize()
//  - can call random_index()
//  - can call pos_from_p() and p_from_pos() if append_prime() has been done

struct indexrange {
    std::vector <index_t>ind;
    std::vector <p_r_values_t>prime;

    /* With -virtual: the ideals of this side are vn->index_of(k, vside)
     * for k below vn->K[vside]. */
    struct virtual_numbering const * vn = nullptr;
    int vside = 0;

    void init() { }
    void finalize() { }

    void append(index_t z) {
        ind.push_back(z);
    }
    void append_prime(p_r_values_t p) {
        prime.push_back(p);
    }

    index_t random_index(index_t z, gmp_randstate_t buf) const {
        // Find position of z
        // Exact version:
        //  vector<index_t>::iterator it;
        //  it = lower_bound (ind.begin(), ind.end(), z);
        //  uint64_t position = it - ind.begin();
        // Fast, approximate version:
        uint64_t const position = z >> 1; // half of indices on each side
        auto const range = uint64_t((double(position))*0.2);
        uint64_t const low = position > range ? position - range : 0;
        uint64_t high = MIN(position + range, vn ? vn->K[vside] : ind.size());
        if (high == low)
            high++;
        if (vn)
            return vn->index_of(low + (u64_random(buf)%(high-low)), vside);
        return ind[low + (u64_random(buf)%(high-low))];
    }

    p_r_values_t p_from_pos(uint64_t position) const {
        return prime[position];
    }

    size_t pos_from_p(p_r_values_t p) const {
        return std::ranges::lower_bound(prime, p) - prime.begin();
    }

    std::vector<index_t> all_composites(
            uint64_t q0,
            uint64_t q1,
            uint64_t qfac_min,
            uint64_t qfac_max,
            int nfactors) const;
    std::vector<std::vector<index_t>> all_composites(
            uint64_t q0,
            uint64_t q1,
            uint64_t qfac_min,
            uint64_t qfac_max) const;
};

// Fill in the indexrange data structure from the renumber table,
// gathering indices in two arrays, one for each side.
// In case of composite special-qs, also fill-in the list of the
// corresponding primes on the sqside.
static std::vector<indexrange> prepare_indexrange(renumber_t const & ren_tab, 
        int sqside, int compsq, int nthreads) {
    std::vector<indexrange> Ind(ren_tab.get_nb_polys());
    for(int side = 0 ; side < ren_tab.get_nb_polys() ; side++)
        Ind[side].init();
    if (!compsq) {
        /* Walking the table takes long when lpb is large, so we split it
         * among the threads: one pass counts the ideals of each side in
         * each thread's slice, and the second one fills the arrays in
         * place. */
        int const nsides = ren_tab.get_nb_polys();
        index_t const N = ren_tab.get_max_index();
        /* floor(N*t/nthreads), without overflow: N = a*nthreads + b */
        auto slice = [&](int t) {
            index_t const a = N / nthreads;
            index_t const b = N % nthreads;
            return a * t + (b * t) / nthreads;
        };
        std::vector<std::vector<uint64_t>> count(nthreads, std::vector<uint64_t>(nsides, 0));
        auto walk = [&](int t, bool fill, std::vector<uint64_t> pos) {
            index_t i = slice(t);
            for (renumber_t::const_iterator it(ren_tab, i) ; i < slice(t + 1) ; ++it, ++i) {
                if (ren_tab.is_additional_column(i))
                    continue;
                int const side = (*it).side;
                if (fill)
                    Ind[side].ind[pos[side]++] = i;
                else
                    count[t][side]++;
            }
        };
        std::vector<std::thread> threads;
        for (int t = 0 ; t < nthreads ; t++)
            threads.emplace_back(walk, t, false, std::vector<uint64_t>());
        for (auto & th : threads) th.join();
        threads.clear();
        std::vector<std::vector<uint64_t>> start(nthreads, std::vector<uint64_t>(nsides, 0));
        for (int side = 0 ; side < nsides ; side++) {
            uint64_t s = 0;
            for (int t = 0 ; t < nthreads ; t++) {
                start[t][side] = s;
                s += count[t][side];
            }
            Ind[side].ind.resize(s);
        }
        for (int t = 0 ; t < nthreads ; t++)
            threads.emplace_back(walk, t, true, start[t]);
        for (auto & th : threads) th.join();
        return Ind;
    }
    index_t i = 0;
    for (auto it = ren_tab.begin() ; it != ren_tab.end() ; ++it, ++i) {
        if (ren_tab.is_additional_column(i))
            continue;
        renumber_t::p_r_side const x = *it;
        Ind[x.side].append(i);
        if (compsq && (x.side == sqside)) {
            Ind[sqside].append_prime(x.p);
        }
    }
    for(int side = 0 ; side < ren_tab.get_nb_polys() ; side++)
        Ind[side].finalize();

    return Ind;
}

static void remove_special_q(relation & rel, special_q const & Q)
{
    auto & V = rel.sides[Q.side];
    auto nn = V.begin();
    for(auto const & pr : V) {
        if (!Q.is_coprime_to(mpz_get_ui(pr.p)))
            continue;
        *nn++ = pr;
    }
    V.erase(nn, V.end());
}


struct model_relation : public indexed_relation_byside {
    template<typename... Args>
        model_relation(Args&&... args)
        : indexed_relation_byside { std::forward<Args>(args)... }
    {}
    /* the counterpart of model_relation(rel, renumber_table), with the
     * ideals numbered by virtual_numbering */
    static model_relation from_virtual(relation const & rel,
            virtual_numbering const & vn)
    {
        model_relation m((relation_ab const &) rel);
        m.set_active_sides(rel.active_sides);
        for (unsigned int side_index = 0 ; side_index < 2 ; side_index++) {
            int const side = rel.active_sides[side_index];
            for (auto const & pr : rel.sides[side]) {
                index_t const i = vn.index(mpz_get_ui(pr.p), side);
                for (int k = pr.e ; k-- ; )
                    m[side_index].push_back(i);
            }
        }
        return m;
    }

    model_relation perturb(std::vector<indexrange> const & Ind, gmp_randstate_t buf) const
    {
        relation_ab const ab(i64_random(buf), u64_random(buf));
        model_relation rel(ab);

        rel.set_active_sides(get_active_sides());
        for(size_t side = 0 ; side < sides.size() ; side++) {
            for(auto i : sides[side]) {
                rel[side].push_back(Ind[side].random_index(i, buf));
            }
        }
        return rel;
    }
};

// static std::map<special_q, std::vector<model_relation> >
static std::pair<std::vector<size_t>, std::vector<model_relation>>
read_sample_file(int sqside, const char *filename,
        std::function<model_relation(relation const &)> const & to_model,
        bool keep_sq)
{
    ifstream_maybe_compressed in(filename);

    ASSERT_ALWAYS (in);

    std::set<special_q> current;
    std::map<special_q, std::vector<model_relation> > sample;

    int nbegin = 0, nend = 0, maxdepth = 0;

    /* With batch cofactorization, las prints the relations at the very
     * end, each group under a line "# <n> relations for <special-q>",
     * when no special-q is being sieved any more. */
    std::optional<special_q> batch_q;

    for(std::string line ; std::getline(in, line) ; ) {
        namespace markers = cado::special_markers;
        if (auto const batch = markers::batch_relations::parse(line)) {
            ASSERT_ALWAYS(sqside == batch->Q.side);
            batch_q = batch->Q;
        } else if (auto const begin = markers::begin_special_q::parse(line)) {
            special_q const & Q = begin->Q;
            ASSERT_ALWAYS(sqside == Q.side);
            sample[Q];  // auto-vivify
            if (current.insert(Q).second) {
                /* When we start over, there's a "second" beginning. */
                nbegin++;
                maxdepth = std::max(maxdepth, nbegin - nend);
            }
        } else if (auto const end = markers::end_special_q::parse(line)) {
            nend++;
            current.erase(end->Q);
        } else if (line.starts_with(markers::begin_special_q::prefix)) {
            throw cado::error("parse error at line: {}", line);
        } else if (line[0] == '#') {
            continue;
        } else {
            relation rel;
            std::istringstream(line) >> rel;
            if (current.empty() && batch_q) {
                if (!keep_sq)
                    remove_special_q(rel, *batch_q);
                sample[*batch_q].push_back(to_model(rel));
                continue;
            }
            for(auto const & Q : current) {
                if (current.size() > 1) {
                    /* If current.size() > 1 it's more difficult, as have to
                     * look up which Q is the good one. There may even be
                     * more than one, but we consider this unlikely here,
                     * since we don't expect more than a few "active"
                     * special-Qs to choose from
                     */

                    /* So do we have a-br = 0 mod p, with the usual
                     * conventions ? Those are actually quite annoying
                     * conventions in general. r represents a point in
                     * P1(Z/p), and we want to check if a:b matches r.
                     *
                     * In practice,
                     *  - Q is always square-free
                     *  - unless allow_compsq is set, Q is prime
                     *  - projective roots above special-q's are not
                     *  used (and this is true even for "partially
                     *  projective" roots over divisors of Q).
                     *
                     * So that it's actually easy, we just have to check
                     * that a-br = 0 mod p
                     */
                    cxx_mpz z;
                    mpz_mul(z, rel.bz, Q.r);
                    mpz_sub(z, rel.az, z);
                    mpz_mod(z, z, Q.p);
                    if (mpz_sgn(z) != 0)
                        continue;
                }
                if (!keep_sq)
                    remove_special_q(rel, Q);
                sample[Q].push_back(to_model(rel));
                break;
            }
        }
    }

    for(auto & S : sample)
        std::ranges::sort(S.second);

    if (nbegin == 0) {
        throw cado::error("# The sample file {} was apparently created without -v, but -v is mandatory for fake_rels",
                filename);
    }

    size_t nq = 0;
    size_t nr = 0;

    // return sample;
    std::pair<std::vector<size_t>, std::vector<model_relation>> ret;
    for(auto const & x : sample) {
        nq++;
        nr+=x.second.size();
        ret.first.push_back(x.second.size());
        std::ranges::copy(x.second, std::back_inserter(ret.second));
    }

    fmt::print("# {}: {} special-q's, {} relations, max {} concurrent special-q's\b",
            filename, nq, nr, maxdepth);
    return ret;
}

static unsigned long print_fake_rel_manyq(
        std::ostream& os,
        std::vector<index_t>::const_iterator qbegin,
        std::vector<index_t>::const_iterator qend,
        int nq,
        // std::vector<std::pair<special_q, std::vector<model_relation> > > const & sample,
        std::pair<std::vector<size_t>, std::vector<model_relation>> const & sample,
        std::vector<indexrange> const & Ind,
        int dl, double shrink_factor,
        gmp_randstate_t buf)
{
    unsigned long nrels_thread = 0;

    auto R = [buf](unsigned long n) { return u64_random(buf) % n; };

    /* nq == 0: each entry stands for a special-q that adds no ideal to
     * the relations (composite special-q, see -chunks) */
    ASSERT_ALWAYS(nq == 0 || (qend - qbegin) % nq == 0);

    /* Each thread writes its relations in chunks of about 1 MB, which
     * still shows progress. Writing after each special-q, as we used
     * to, meant one locked write per relation or so: with many threads,
     * they all queued on the lock while one of them wrote a hundred
     * bytes to the output. */
    std::ostringstream oss;
    auto flush = [&]() {
        std::lock_guard<std::mutex> const dummy(io_mutex);
        os << oss.str();
        oss.str("");
    };

    for (auto it = qbegin ; it != qend ; ) {
        /* This is the part that will go in _all_ relations */
        std::vector<index_t> qpart;
        if (nq == 0)
            ++it;
        for(int n = nq ; n-- ; )
            qpart.push_back(*it++);

        auto const & model_nrels = sample.first[R(sample.first.size())];
        // special_q const & model_q = model.first;
        // auto const & model_nrels = model.second.size();

        int nr;
        if (shrink_factor == 1) {
            nr = model_nrels;
        } else {
            double const nr_dble = double_ratio(model_nrels, shrink_factor);
            // Do probabilistic rounding, in case nr_dble is small (maybe < 1)
            double const trunc_part = trunc(nr_dble);
            double const frac_part = nr_dble - trunc_part;
            double const rnd = double_ratio(u64_random(buf), UINT64_MAX);
            nr = int(trunc_part) + int(rnd < frac_part);
        }

        for( ; nr-- ; ) {
            // auto const & model_rel = model.second[R(model_nrels)];
            auto const & model_rel = sample.second[R(sample.second.size())];

            model_relation rel = model_rel.perturb(Ind, buf);

            /* Note that we always add the indices that correspond to q
             * to the **END** of the relation, irrespective of which side
             * q is on ! This is because:
             *   - it doesn't make any difference * down the line,
             *   - and keeping track of the proper side for q (or,
             *   conceivably, for the sides of all divisors of q !) would
             *   be a bit annoying here.
             */
            std::ranges::copy(qpart, std::back_inserter(rel.sides.back()));

            if (shrink_factor > 1)
                rel.shrink(shrink_factor);
            rel.sort();
            rel.compress(dl);

            oss << rel << "\n";
            nrels_thread++;
            // rels_printed++;
        }
        if (oss.tellp() > (1 << 20))
            flush();
    }
    flush();
    return nrels_thread;
}

std::vector<index_t> indexrange::all_composites(uint64_t q0, uint64_t q1,
        uint64_t qfac_min,
        uint64_t qfac_max,
        int n) const
{
    std::vector<index_t> ret;

    if (n == 0)
        return ret;

    auto jt = std::back_inserter(ret);

    /* primes divisors of n-factor composite special-qs can never be
     * smaller than this:
     */
    uint64_t l1min = MAX(q0/pow(qfac_max, n-1), qfac_min);
    /* XXX prime at this position might be below l1min ! */
    uint64_t pos_l1min = pos_from_p(l1min);
    for( ; p_from_pos(pos_l1min) < l1min ; pos_l1min++);
    l1min = p_from_pos(pos_l1min);

    /* and never bigger than this: */
    uint64_t const l1max = MIN(qfac_max, round(pow(q1, 1/(double) n)));
    uint64_t const pos_l1max = pos_from_p(l1max);

    for (uint64_t pos1 = pos_l1min; pos1 < pos_l1max; ++pos1) {
        /* look for cases where _this_ prime is the smallest one */
        uint64_t const l1 = p_from_pos(pos1);
        /* q0 <= l1 * x < q1
         * implies q0/l1 <= x < q1/l1
         * the left part is easy, but for the right part, we rewrite as:
         * l1 * x <= q1-1
         * x <= (q1-1)/l1 ---> x <= floor((q1-1)/l1)  --> x < ceil(q1/l1)
         */
        if (n == 1) {
            /* no point in recursing to determine a list of zero-length
             * continuations.
             */
            *jt++ = pos1;
        } else {
            auto tail = all_composites(iceildiv(q0,l1), iceildiv(q1,l1), l1, qfac_max, n-1);
            for(auto it = tail.begin() ; it != tail.end() ; ) {
                *jt++ = pos1;
                for(int j=n-1 ; j-- ; )
                    *jt++ = *it++;
            }
        }
    }

    return ret;
}

std::vector<std::vector<index_t>> indexrange::all_composites(uint64_t q0, uint64_t q1,
        uint64_t qfac_min,
        uint64_t qfac_max) const
{
    std::vector<std::vector<index_t>> list;
    list.emplace_back();
    list.emplace_back();
    for(int n = 2 ; ; n++) {
        list.emplace_back(all_composites(q0, q1, qfac_min, qfac_max, n));
        if (list.back().empty()) {
            list.pop_back();
            break;
        }
        fmt::print("# Got {} {}-composite sq\n",
                (size_t)(list.back().size()/n), n);
    }
    return list;
}

/* With -virtual, the special-q of [q0,q1) on side sqside are the ideals
 * vn->index_of(k, sqside) for k in [vk0, vk1). */
struct virtual_qrange {
    virtual_numbering const * vn = nullptr;
    uint64_t vk0 = 0, vk1 = 0;
    int sqside = 0;
    /* composite special-q (a -chunks line with nsq): nsq special-q that
     * add no ideal to the relations */
    bool no_q = false;
    uint64_t nsq = 0;
};

static void worker(int tnum, int nt,
        std::ostream & os,
        std::vector<indexrange> const & Ind,
        // std::vector<std::pair<special_q, std::vector<model_relation>>> const & sample,
        std::pair<std::vector<size_t>, std::vector<model_relation>> const & sample,
        std::vector<std::vector<index_t>> const & qs,
        virtual_qrange const & vq,
        double shrink_factor, int dl, unsigned long seed, uint64_t q0)
{
    cxx_gmp_randstate buf;
    /* The stream must depend on the q range, not only on the thread:
     * two runs (or two -chunks) that drew from the same sample with the
     * same streams would print the same relations, up to the special-q,
     * and these near-copies make the matrix look much easier to get
     * than it is. The seed has 64 bits, which an unsigned long may not
     * hold. */
    uint64_t const s = seed + tnum + q0 * UINT64_C(0x9E3779B97F4A7C15)
        + (vq.no_q ? UINT64_C(0x632BE59BD9B4E019) : 0);
    gmp_randseed(buf, cxx_mpz(s));
    unsigned long ret = 0;
    if (vq.no_q) {
        uint64_t const k0 = (tnum * vq.nsq) / nt;
        uint64_t const k1 = ((tnum + 1) * vq.nsq) / nt;
        std::vector<index_t> const block(1U << 16, 0);
        for (uint64_t k = k0 ; k < k1 ; ) {
            uint64_t const n = std::min<uint64_t>(k1 - k, block.size());
            ret += print_fake_rel_manyq(os, block.begin(), block.begin() + n,
                    0, sample, Ind, dl, shrink_factor, buf);
            k += n;
        }
    } else if (vq.vn) {
        uint64_t const n = vq.vk1 - vq.vk0;
        uint64_t const k0 = vq.vk0 + (tnum * n) / nt;
        uint64_t const k1 = vq.vk0 + ((tnum + 1) * n) / nt;
        std::vector<index_t> block;
        for (uint64_t k = k0 ; k < k1 ; ) {
            block.clear();
            for ( ; k < k1 && block.size() < (1U << 16) ; k++)
                block.push_back(vq.vn->index_of(k, vq.sqside));
            ret += print_fake_rel_manyq(os, block.begin(), block.end(),
                    1, sample, Ind, dl, shrink_factor, buf);
        }
    }
    for(size_t n = 1 ; n < qs.size() ; n++) {
        ASSERT_ALWAYS(qs[n].size() % n == 0);
        auto it0 = qs[n].begin() + n * ((tnum * qs[n].size() / n) / nt);
        auto it1 = qs[n].begin() + n * (((tnum + 1) * qs[n].size() / n) / nt);
        ret += print_fake_rel_manyq(
                os,
                it0, it1,
                n,
                sample,
                Ind,
                dl, shrink_factor,
                buf);
    }
    std::lock_guard<std::mutex> const dummy(io_mutex);
    rels_printed += ret;
}

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage("poly", "polynomial file");
    pl.declare_usage("lpb0", "large prime bound on side 0");
    pl.declare_usage("lpb1", "large prime bound on side 1");
    pl.declare_usage("q0", "lower bound of the qrange");
    pl.declare_usage("q1", "upper bound of the qrange");
    pl.declare_usage("sqside", "side of the special-q");
    pl.declare_usage("sample", "file where to find a sample of relations");
    pl.declare_usage("chunks", "file of lines \"q0 q1 sample output [nsq]\":"
            " do each of these q ranges, with its sample, into its own output"
            " file, in one run (instead of -q0, -q1, -sample; the renumber"
            " table is then read once). With nsq, the special-q of that line"
            " are composite, made of factor base primes (-allow-compsq in"
            " las, with qfac-max below lim), and there are nsq of them: the"
            " primes of the special-q stay in the sample relations, perturbed"
            " as the other ideals, and no special-q ideal is added");
    pl.declare_usage("renumber", "renumber table");
    pl.declare_usage("virtual", "(switch) no renumber table: number the ideals"
            " by the count of primes below their norm (see virtual_numbering);"
            " for parameters whose table would not fit");
    pl.declare_usage("shrink-factor", "simulate with a matrix that number (integer >= 1) times smaller");
    pl.declare_usage("dl", "dl mode");
    pl.declare_usage("allow-compsq", "(switch) allows composite sq");
    pl.declare_usage("qfac-min", "factors of q must be at least that");
    pl.declare_usage("qfac-max", "factors of q must be at most that");
    pl.declare_usage("seed", "random seed");
    pl.declare_usage("t", "number of threads to use");
    pl.declare_usage("v", "verbose mode");
    verbose_decl_usage(pl);
}

static int main_(int argc, char const * argv[]);

// coverity[root_function]
int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;
    cxx_cado_poly cpoly;
    int sqside = -1;
    int lpb[2] = {0, 0};
    uint64_t q0 = 0;
    uint64_t q1 = 0;
    int dl = 0;
    int mt = 1;
    int compsq = 0;
    int virt = 0;
    uint64_t qfac_min = 1024;
    uint64_t qfac_max = UINT64_MAX;
    double shrink_factor = 1; // by default, no shrink

    declare_usage(pl);
    pl.configure_switch_old("-v", &verbose);
    pl.configure_switch_old("-dl", &dl);
    pl.configure_switch_old("-allow-compsq", &compsq);
    pl.configure_switch_old("-virtual", &virt);

    argv++, argc--;
    for( ; argc ; ) {
        if (pl.update_cmdline(argc, argv)) { continue; }

        /* Could also be a file */
        FILE *f;
        if ((f = fopen(argv[0], "r")) != nullptr) {
            pl.read(f, false);
            fclose(f);
            argv++,argc--;
            continue;
        }

        pl.fail("Unhandled parameter {}", argv[0]);
    }
    verbose_interpret_parameters(pl);
    pl.print_command_line(stdout);

    const char * filename = pl.lookup_old("poly");
    if (!filename) {
        pl.fail("Error: parameter -poly is mandatory");
    }

    pl.parse_mandatory("lpb0", lpb[0]);
    pl.parse_mandatory("lpb1", lpb[1]);
    struct chunk {
        uint64_t q0, q1;
        std::string sample, output;
        uint64_t nsq = 0;
    };
    std::vector<chunk> chunks;
    if (const char * chunkfile = pl.lookup_old("chunks")) {
        std::ifstream in(chunkfile);
        if (!in)
            pl.fail("Error: cannot read {}", chunkfile);
        for (std::string line ; std::getline(in, line) ; ) {
            if (line.empty() || line[0] == '#')
                continue;
            chunk c;
            std::istringstream is(line);
            if (!(is >> c.q0 >> c.q1 >> c.sample >> c.output))
                pl.fail("Error: bad line in {}: {}", chunkfile, line);
            is >> c.nsq;    /* optional: composite special-q */
            chunks.push_back(c);
        }
    } else {
        pl.parse_mandatory("q0", q0);
        pl.parse_mandatory("q1", q1);
        const char * samplefile = pl.lookup_old("sample");
        if (!samplefile) {
            pl.fail("Error: parameter -sample is mandatory");
        }
        chunks.push_back({ q0, q1, samplefile, {} });
    }

    pl.parse("shrink-factor", shrink_factor);
    if (shrink_factor < 1) {
        pl.fail("Error: shrink factor must be an integer >= 1");
    }

    pl.parse("t", mt);
    pl.parse("qfac-min", qfac_min);
    pl.parse("qfac-max", qfac_max);

    unsigned long seed = 171717;
    pl.parse("seed", seed);

    if (!cpoly.read(filename))
    {
        throw cado::error("Error reading polynomial file {}", filename);
    }

    pl.parse("sqside", sqside);
    if (sqside < 0 || sqside > 1) {
        pl.fail("Error: sqside must be 0 or 1");
    }

    renumber_t ren_table(cpoly);
    std::unique_ptr<virtual_numbering> vn;
    if (virt) {
        if (compsq)
            pl.fail("Error: -virtual does not support composite special-q");
        vn = std::make_unique<virtual_numbering>(std::array<int, 2> { lpb[0], lpb[1] });
    } else {
        const char * renumberfile = pl.lookup_old("renumber");
        if (!renumberfile) {
            pl.fail("Error: parameter -renumber is mandatory");
        }
        fmt::print ("# Start reading renumber table\n");
        fflush (stdout);
        ren_table.read_from_file(renumberfile, dl);
        fmt::print ("# Done reading renumber table\n");
        fflush (stdout);

        for (int side = 0; side < ren_table.get_nb_polys(); ++side) {
            if (ren_table.get_lpb(side) != (unsigned long)lpb[side]) {
                pl.fail("Error: on side {}, lpb on the command-line is different from the one in the renumber file",
                        side);
            }
        }
    }

    pl.warn_unused();

    // Two index ranges, one for each side
    fmt::print ("# Start preparing index ranges\n");
    fflush (stdout);
    std::vector<indexrange> Ind;
    if (virt) {
        Ind.resize(2);
        for (int side = 0 ; side < 2 ; side++) {
            Ind[side].vn = vn.get();
            Ind[side].vside = side;
        }
    } else {
        Ind = prepare_indexrange(ren_table, sqside, compsq, mt);
    }
    fmt::print ("# Done preparing index ranges\n");
    fflush (stdout);

    std::function<model_relation(relation const &)> to_model;
    if (virt) {
        to_model = [&vn](relation const & rel) {
            return model_relation::from_virtual(rel, *vn);
        };
    } else {
        to_model = [&ren_table](relation const & rel) {
            return model_relation(rel, ren_table);
        };
    }

    for (auto const & C : chunks) {
        q0 = C.q0;
        q1 = C.q1;

        fmt::print ("# Start reading sample file\n");
        fflush (stdout);

        std::pair<std::vector<size_t>, std::vector<model_relation>> const sample = read_sample_file(sqside, C.sample.c_str(), to_model, C.nsq > 0);

        fmt::print ("# Done reading sample file\n");
        fflush (stdout);

        std::vector<std::vector<index_t>> qs;

        /* the indexranges contain the prime-to-index conversion functionality
         * only if compsq holds. In the non-composite case, we fill qs[1]
         * differently, because we don't want to 
         */
        virtual_qrange vq;
        if (C.nsq) {
            vq.no_q = true;
            vq.nsq = C.nsq;
            fmt::print("# {} composite special-q, no special-q ideal\n", vq.nsq);
        } else if (virt) {
            vq.vn = vn.get();
            vq.vk0 = vn->N(q0);
            vq.vk1 = vn->N(q1);
            vq.sqside = sqside;
            fmt::print("# {} special-q (virtual numbering)\n", vq.vk1 - vq.vk0);
        } else if (!compsq) {
            qs.emplace_back(); /* 0 -- placeholder */
            qs.emplace_back(); /* 1 */

            /* jump-start in the renumber table, and enumerate the large primes
             * from there.
             */
            index_t i = ren_table.index_from_p(q0, sqside);
            renumber_t::const_iterator it(ren_table, i);
            for( ; it != ren_table.end() && (*it).p < q1 ; ++it, ++i) {
                if ((*it).side == sqside)
                    qs[1].push_back(i);
            }
        } else {
            /* In theory, this is overkill. We don't _really_ need the prime[]
             * cache in the indexrange type. Except that the renumber
             * "traditional" format has expensive *_from_index lookups, and
             * that would get in the way if we want to do away with
             * indexrange::prime[]. Also, we don't have bidirectional iterators
             * on the renumber table, so it's not really practical to use only
             * the renumber table.
             */
            qs = Ind[sqside].all_composites(q0, q1, qfac_min, qfac_max);
            /* all_composites gives positions in the list of the ideals
             * of that side; the relations want their indices */
            for (auto & v : qs)
                for (auto & x : v)
                    x = Ind[sqside].ind[x];
        }

        std::ofstream out;
        if (!C.output.empty()) {
            out.open(C.output);
            if (!out)
                throw cado::error("cannot write to {}", C.output);
        }
        std::ostream & os = C.output.empty() ? std::cout : out;

        std::vector<std::thread> threads;

        double t0 = seconds ();
        double wct_t0 = wct_seconds ();
        rels_printed = 0;

        threads.reserve(mt);
        /* std::thread copies its arguments unless told otherwise: without
         * std::cref, each thread got its own copy of the index ranges, of
         * the special-q list and of the sample, which takes gigabytes and
         * minutes per thread at record sizes. */
        for(int i = 0 ; i < mt ; i++)
            threads.emplace_back(worker, i, mt, std::ref(os),
                    std::cref(Ind), std::cref(sample), std::cref(qs),
                    std::cref(vq), shrink_factor, dl, seed, q0);
        for(auto & t : threads)
            t.join();

        if (!C.output.empty()) {
            out.close();
            if (!out)
                throw cado::error("error writing {}", C.output);
        }

        t0 = seconds () - t0;
        wct_t0 = wct_seconds () - wct_t0;

        /* print statistics */

        fmt::print ("# Output {} relations in {:.2f}s cpu ({:.0f} rels/s)\n",
                rels_printed, t0, double_ratio(rels_printed, t0));
        fmt::print ("# Output {} relations in {:.2f}s wct ({:.0f} rels/s)\n",
                rels_printed, wct_t0, double_ratio(rels_printed, wct_t0));
        if (!C.output.empty())
            fmt::print ("# q in [{}, {}), sample {}: {} relations to {}\n",
                    q0, q1, C.sample, rels_printed, C.output);
        fflush (stdout);
    }

    return 0;
}
