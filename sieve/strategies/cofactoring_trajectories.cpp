#include "cado.h" // IWYU pragma: keep

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "cado_main.hpp"
#include "cxx_mpz.hpp"
#include "facul.hpp"
#include "facul_method.hpp"
#include "facul_strategies.hpp"
#include "factorization_patterns.hpp"
#include "gmp_aux.h"
#include "random_distributions.hpp"
#include "macros.h"
#include "modset.hpp"
#include "params.hpp"
#include "utils_cxx.hpp"

/* Run the chain of the default strategy on each side of each pair that
 * las -dump-cofactors wrote, side by side independently, and record what
 * happens at each step: where the side first splits, what that split
 * says, and how the remaining composites (if any) are finished off by
 * the methods after it. This is enough to work out, for any number of
 * methods on each side and any interleaving of the two chains, both
 * whether facul_all() would find the relation and what it would cost.
 *
 * Output, one line per pair:
 *   r0 r1 j0 o0 e0 x0 [t...] ; j1 o1 e1 x1 [t...]
 * where for each side, j is the 1-based index of the method that first
 * splits it (0 if none does within the chain), o is T (nothing to do:
 * 1 or a prime below lim^2), N (the split shows it is not smooth), S (the
 * split completes the factorization), A (a composite is left), U (no
 * split); for A, e is the index of the method at which the remaining
 * composites are all dealt with (0 if they never are), x is S or N, and
 * the t are the cumulative times (in ns) of the aux steps j+1..e.
 *
 * The mean time of each method of the chain, per side and cofactor size,
 * goes to the file given by -times.
 */

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage("in", "file of pairs, one \"a b N0 N1\" per line");
    pl.declare_usage("synthetic", "instead of -in: this many random cofactors per side and size, drawn from the factorization patterns that list_factorization_patterns gives (the other side being 1)");
    pl.declare_usage("seed", "seed for -synthetic (default 0)");
    pl.declare_usage("out", "output file");
    pl.declare_usage("times", "output file for mean method times");
    pl.declare_usage("lim0", "factor base bound on side 0");
    pl.declare_usage("lim1", "factor base bound on side 1");
    pl.declare_usage("lpb0", "large prime bound on side 0, in bits");
    pl.declare_usage("lpb1", "large prime bound on side 1, in bits");
    pl.declare_usage("mfb0", "cofactor bound on side 0, in bits; larger pairs are skipped");
    pl.declare_usage("mfb1", "cofactor bound on side 1, in bits; larger pairs are skipped");
    pl.declare_usage("ncurves", "length of the default chain, in curves after the first ones");
    pl.declare_usage("t", "number of threads (default 1)");
}

namespace {

uint64_t thread_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return uint64_t(ts.tv_sec) * 1000000000UL + uint64_t(ts.tv_nsec);
}

struct side_trajectory {
    unsigned int j = 0;
    char o = 'U';
    unsigned int e = 0;
    char x = '-';
    std::vector<uint64_t> aux_times;
};

struct time_acc {
    std::vector<std::vector<double>> sum;       /* [r][k] */
    std::vector<std::vector<unsigned long>> n;
    time_acc(unsigned int mfb, size_t K)
        : sum(mfb + 1, std::vector<double>(K, 0))
        , n(mfb + 1, std::vector<unsigned long>(K, 0))
    {}
};

side_trajectory run_side(cxx_mpz const & N,
        std::vector<facul_method> const & chain,
        unsigned long lpb, double BB, double BBB, time_acc & acc)
{
    side_trajectory T;
    if (N == 1 || mpz_get_d(N) < BB) {
        T.o = 'T';
        return T;
    }
    std::unique_ptr<FaculModulusBase> n(FaculModulusBase::init_mpz(N));
    ASSERT_ALWAYS(n);
    unsigned int const r = N.bits();

    std::vector<std::unique_ptr<FaculModulusBase>> todo;
    size_t k = 0;
    for ( ; k < chain.size() ; k++) {
        std::vector<cxx_mpz> primes;
        std::vector<std::unique_ptr<FaculModulusBase>> comps;
        uint64_t const t0 = thread_ns();
        facul_status const st = n->facul_doit_onefm(primes, chain[k], comps, lpb, BB, BBB);
        uint64_t const t1 = thread_ns();
        acc.sum[r][k] += double(t1 - t0);
        acc.n[r][k]++;
        if (st == FACUL_NOT_SMOOTH) {
            T.j = k + 1; T.o = 'N';
            return T;
        }
        if (st == FACUL_SMOOTH) {
            T.j = k + 1; T.o = 'S';
            return T;
        }
        if (!comps.empty()) {
            T.j = k + 1; T.o = 'A';
            todo = std::move(comps);
            break;
        }
        /* a prime found without any composite would have been SMOOTH or
         * NOT_SMOOTH: so here, nothing was found */
        ASSERT_ALWAYS(primes.empty());
    }
    if (T.o != 'A')
        return T;

    /* this is facul_aux */
    uint64_t cumul = 0;
    for (k++ ; k < chain.size() && !todo.empty() ; k++) {
        std::vector<std::unique_ptr<FaculModulusBase>> next_todo;
        uint64_t const t0 = thread_ns();
        bool ns = false;
        for (auto & c : todo) {
            std::vector<cxx_mpz> primes;
            std::vector<std::unique_ptr<FaculModulusBase>> more;
            facul_status const st = c->facul_doit_onefm(primes, chain[k], more, lpb, BB, BBB);
            if (st == FACUL_NOT_SMOOTH) { ns = true; break; }
            if (st == FACUL_MAYBE) {
                if (more.empty())
                    next_todo.emplace_back(std::move(c));
                else
                    for (auto & m : more)
                        next_todo.emplace_back(std::move(m));
            }
        }
        cumul += thread_ns() - t0;
        T.aux_times.push_back(cumul);
        if (ns) {
            T.e = k + 1; T.x = 'N';
            return T;
        }
        std::swap(todo, next_todo);
    }
    if (todo.empty()) {
        T.e = k; T.x = 'S';
    }
    return T;
}

void print_side(std::ostream & os, side_trajectory const & T)
{
    os << T.j << " " << T.o << " " << T.e << " " << T.x;
    for (auto t : T.aux_times)
        os << " " << t;
}

/* Cofactors as las hands them over, for each side and size: random
 * products of primes above lim, along the factorization patterns of that
 * size, each drawn with the weight that list_factorization_patterns
 * estimates (las drops the cofactors that are prime, so the patterns
 * have two factors or more); and one cofactor per size up to lpb, which
 * needs nothing. The other side of each pair is 1. Each (side, size) has
 * its own random state, so that the result does not depend on the
 * number of threads. */
std::vector<std::vector<cxx_mpz>> synthetic_pairs(unsigned long const * lim,
        unsigned int const * lpb, unsigned int const * mfb,
        unsigned long n, unsigned long seed, unsigned int nthreads)
{
    struct task { int s; unsigned int r; std::vector<std::vector<cxx_mpz>> out; };
    std::vector<task> tasks;
    for (int s = 0 ; s < 2 ; s++)
        for (unsigned int r = lpb[s] + 1 ; r <= mfb[s] ; r++)
            tasks.push_back({ s, r, {} });
    auto worker = [&](unsigned int t) {
        for (size_t i = t ; i < tasks.size() ; i += nthreads) {
            auto & T = tasks[i];
            auto const P = list_factorization_patterns(T.r, lim[T.s]);
            if (P.empty())
                continue;
            std::vector<double> pmf;
            double total = 0;
            for (auto const & p: P)
                total += p.count;
            for (auto const & p: P)
                pmf.push_back(p.count / total);
            cxx_gmp_randstate state;
            gmp_randseed_ui(state, seed * 1000003UL + T.s * 1009UL + T.r);
            auto prime = [&](unsigned int b) {
                cxx_mpz p;
                for (;;) {
                    mpz_urandomb(p, state, b - 1);
                    mpz_setbit(p, b - 1);
                    mpz_nextprime(p, p);
                    if (mpz_sizeinbase(p, 2) == b && mpz_cmp_ui(p, lim[T.s]) > 0)
                        return p;
                }
            };
            while (T.out.size() < n) {
                auto const & pat = P[random_categorical(state, pmf)];
                for (int tries = 0 ; tries < 50 ; tries++) {
                    cxx_mpz N = 1;
                    for (auto const b: pat)
                        mpz_mul(N, N, prime(b));
                    if (N.bits() == T.r) {
                        std::vector<cxx_mpz> pair(2, cxx_mpz(1));
                        pair[T.s] = N;
                        T.out.push_back(std::move(pair));
                        break;
                    }
                }
            }
        }
    };
    std::vector<std::thread> threads;
    for (unsigned int t = 0 ; t < nthreads ; t++)
        threads.emplace_back(worker, t);
    for (auto & th: threads)
        th.join();
    std::vector<std::vector<cxx_mpz>> res;
    for (int s = 0 ; s < 2 ; s++) {
        for (unsigned int r = 2 ; r <= lpb[s] ; r++) {
            std::vector<cxx_mpz> pair(2, cxx_mpz(1));
            mpz_setbit(pair[s], r - 1);
            res.push_back(std::move(pair));
        }
    }
    for (auto & T: tasks)
        for (auto & pair: T.out)
            res.push_back(std::move(pair));
    return res;
}

} /* namespace */

static int main_(int argc, char const * argv[]);

int main(int argc, char const * argv[])
{
    return cado::main_wrapper(main_, argc, argv);
}

static int main_(int argc, char const * argv[])
{
    cxx_param_list pl;
    declare_usage(pl);
    pl.process_command_line_and_extra_parameter_files(argc, argv);

    std::string infile;
    pl.parse("in", infile);
    unsigned long nsynth = 0;
    pl.parse("synthetic", nsynth);
    unsigned long seed = 0;
    pl.parse("seed", seed);
    if (infile.empty() == (nsynth == 0))
        pl.fail("give either -in or -synthetic");
    auto const outfile = pl.parse_mandatory<std::string>("out");
    auto const timesfile = pl.parse_mandatory<std::string>("times");
    unsigned long const lim[2] = {
        pl.parse_mandatory<unsigned long>("lim0"),
        pl.parse_mandatory<unsigned long>("lim1") };
    unsigned int const lpb[2] = {
        pl.parse_mandatory<unsigned int>("lpb0"),
        pl.parse_mandatory<unsigned int>("lpb1") };
    unsigned int const mfb[2] = {
        pl.parse_mandatory<unsigned int>("mfb0"),
        pl.parse_mandatory<unsigned int>("mfb1") };
    int const ncurves = pl.parse_mandatory<int>("ncurves");
    unsigned int nthreads = 1;
    pl.parse("t", nthreads);
    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    std::vector<facul_method> chain;
    for (auto const & mp : facul_strategy_oneside::default_strategy(ncurves))
        chain.emplace_back(mp);

    std::vector<std::vector<cxx_mpz>> pairs;
    if (nsynth)
        pairs = synthetic_pairs(lim, lpb, mfb, nsynth, seed, nthreads);
    else {
        std::ifstream is(infile);
        if (!is)
            throw cado::error("cannot read {}", infile);
        std::string line;
        while (std::getline(is, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream ls(line);
            int64_t a;
            uint64_t b;
            std::vector<cxx_mpz> n(2);
            if (!(ls >> a >> b >> n[0] >> n[1]))
                throw cado::error("cannot parse line: {}", line);
            if (n[0].bits() > mfb[0] || n[1].bits() > mfb[1])
                continue;
            pairs.emplace_back(std::move(n));
        }
    }

    double const BB[2] = { double(lim[0]) * double(lim[0]), double(lim[1]) * double(lim[1]) };
    double const BBB[2] = { BB[0] * double(lim[0]), BB[1] * double(lim[1]) };

    std::vector<std::string> lines(pairs.size());
    std::vector<std::vector<time_acc>> acc;
    for (unsigned int t = 0 ; t < nthreads ; t++)
        acc.push_back({ time_acc(mfb[0], chain.size()), time_acc(mfb[1], chain.size()) });

    auto worker = [&](unsigned int t) {
        for (size_t i = t; i < pairs.size(); i += nthreads) {
            std::ostringstream os;
            os << pairs[i][0].bits() << " " << pairs[i][1].bits() << " ";
            for (int s = 0 ; s < 2 ; s++) {
                auto T = run_side(pairs[i][s], chain, lpb[s], BB[s], BBB[s], acc[t][s]);
                if (s) os << " ; ";
                print_side(os, T);
            }
            lines[i] = os.str();
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < nthreads; t++)
        threads.emplace_back(worker, t);
    for (auto & th: threads)
        th.join();

    {
        std::ofstream os(outfile);
        for (auto const & l : lines)
            os << l << "\n";
    }
    {
        std::ofstream os(timesfile);
        os << "# side r k mean_ns count\n";
        for (int s = 0 ; s < 2 ; s++) {
            for (unsigned int r = 0 ; r <= mfb[s] ; r++) {
                for (size_t k = 0 ; k < chain.size() ; k++) {
                    double sum = 0;
                    unsigned long n = 0;
                    for (auto const & a : acc) {
                        sum += a[s].sum[r][k];
                        n += a[s].n[r][k];
                    }
                    if (n)
                        os << fmt::format("{} {} {} {:.1f} {}\n", s, r, k + 1, sum / double(n), n);
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
