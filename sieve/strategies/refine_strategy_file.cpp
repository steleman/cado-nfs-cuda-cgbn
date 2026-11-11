#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "cado_main.hpp"
#include "cxx_mpz.hpp"
#include "facul.hpp"
#include "facul_method.hpp"
#include "facul_strategies.hpp"
#include "macros.h"
#include "params.hpp"
#include "utils_cxx.hpp"

/* Improve a strategy file against pairs of cofactors that las wrote with
 * -dump-cofactors, by replaying the cofactorization on them.
 *
 * -reorder: in each cell, the methods of the two sides can run in any
 * interleaving that keeps the order of each side; which pairs are found
 * does not depend on it, only the time does. Each side's chain is cut
 * in a head (up to the method that carries a STOP, or else before the
 * first MPQS) and a tail, and six orderings are tried:
 *   1: c0 c1   2: c1 c0   3: h0 h1 t0 t1   4: h1 h0 t1 t0
 *   5: h0 h1 t1 t0   6: h1 h0 t0 t1
 * Each cell keeps the one that took the least time on its pairs.
 *
 * -prune X: drop the cells whose cofactorization time per relation,
 * summed over the cells within -radius in both sizes (one cell is too
 * noisy), exceeds X times the cpu time per relation of the whole run,
 * C0 (what it took to produce the pairs) plus the replay.
 */

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header("Improve a strategy file by replaying the cofactorization\n"
                            "on the pairs that las -dump-cofactors wrote.\n");
    pl.declare_usage("in", "file of pairs, one \"a b N0 N1\" per line");
    pl.declare_usage("lim0", "factor base bound on side 0");
    pl.declare_usage("lim1", "factor base bound on side 1");
    pl.declare_usage("lpb0", "large prime bound on side 0, in bits");
    pl.declare_usage("lpb1", "large prime bound on side 1, in bits");
    pl.declare_usage("mfb0", "cofactor bound on side 0, in bits; larger pairs are skipped");
    pl.declare_usage("mfb1", "cofactor bound on side 1, in bits; larger pairs are skipped");
    pl.declare_usage("file-cofact", "strategy file to improve");
    pl.declare_usage("out", "strategy file to write");
    pl.declare_usage("reorder", "pick the fastest ordering of the two sides in each cell");
    pl.declare_usage("prune", "drop the cells that cost more than this many times the average time per relation");
    pl.declare_usage("C0", "cpu seconds that it took to produce the pairs (for -prune)");
    pl.declare_usage("radius", "cells within this distance count for -prune (default 2)");
    pl.declare_usage("t", "number of threads (default 1)");
    pl.declare_usage("cells", "write \"r0 r1 calls smooth time\" for each cell with pairs, under the output");
}

namespace {

using chain_t = std::vector<facul_method::parameters_with_side>;
using strategy_file = facul_strategies::strategy_file;
using sizes_t = std::vector<unsigned int>;

struct cell_stats {
    unsigned long calls = 0;
    unsigned long smooth = 0;
    double time = 0;
};

double thread_cputime()
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}

struct replayer {
    std::vector<unsigned long> lim;
    std::vector<unsigned int> lpb;
    std::vector<unsigned int> mfb;
    unsigned int nthreads = 1;
    std::vector<std::vector<cxx_mpz>> pairs;

    /* the statistics of each cell, under the strategies of S */
    std::map<sizes_t, cell_stats> operator()(strategy_file const & S) const
    {
        facul_strategies const F(lim, lpb, mfb, true, S, 0);
        std::vector<std::map<sizes_t, cell_stats>> per_thread(nthreads);
        auto worker = [&](unsigned int t) {
            auto & cells = per_thread[t];
            for (size_t i = t ; i < pairs.size() ; i += nthreads) {
                auto const & n = pairs[i];
                double const t0 = thread_cputime();
                auto res = facul_all(n, F);
                double const t1 = thread_cputime();
                bool ok = true;
                for (auto const & r: res)
                    ok = ok && r.status == FACUL_SMOOTH;
                auto & c = cells[{ unsigned(n[0].bits()), unsigned(n[1].bits()) }];
                c.calls++;
                c.smooth += ok;
                c.time += t1 - t0;
            }
        };
        std::vector<std::thread> threads;
        for (unsigned int t = 0 ; t < nthreads ; t++)
            threads.emplace_back(worker, t);
        for (auto & th: threads)
            th.join();
        std::map<sizes_t, cell_stats> total;
        for (auto const & cells: per_thread) {
            for (auto const & [k, c]: cells) {
                auto & T = total[k];
                T.calls += c.calls;
                T.smooth += c.smooth;
                T.time += c.time;
            }
        }
        return total;
    }
};

/* the head of one side's chain: up to the method that carries a STOP,
 * or else up to the first MPQS, or else everything */
std::pair<chain_t, chain_t> head_and_tail(chain_t const & c)
{
    auto it = std::ranges::find_if(c, [](auto const & m) { return m.stop_unsplit; });
    if (it != c.end())
        ++it;
    else
        it = std::ranges::find_if(c, [](auto const & m) { return m.method == MPQS_METHOD; });
    return { chain_t(c.begin(), it), chain_t(it, c.end()) };
}

chain_t ordering(chain_t const & chain, int o)
{
    std::array<chain_t, 2> c;
    for (auto const & m: chain)
        c[m.side].push_back(m);
    auto [h0, t0] = head_and_tail(c[0]);
    auto [h1, t1] = head_and_tail(c[1]);
    std::vector<chain_t const *> parts;
    switch (o) {
        case 1: parts = { &c[0], &c[1] }; break;
        case 2: parts = { &c[1], &c[0] }; break;
        case 3: parts = { &h0, &h1, &t0, &t1 }; break;
        case 4: parts = { &h1, &h0, &t1, &t0 }; break;
        case 5: parts = { &h0, &h1, &t1, &t0 }; break;
        case 6: parts = { &h1, &h0, &t0, &t1 }; break;
        default: ASSERT_ALWAYS(0);
    }
    chain_t res;
    for (auto const * p: parts)
        res.insert(res.end(), p->begin(), p->end());
    return res;
}

double total_time(std::map<sizes_t, cell_stats> const & stats)
{
    double t = 0;
    for (auto const & [k, c]: stats)
        t += c.time;
    return t;
}

unsigned long total_smooth(std::map<sizes_t, cell_stats> const & stats)
{
    unsigned long s = 0;
    for (auto const & [k, c]: stats)
        s += c.smooth;
    return s;
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
    pl.configure_switch("reorder");
    pl.process_command_line_and_extra_parameter_files(argc, argv);

    replayer R;
    auto const infile = pl.parse_mandatory<std::string>("in");
    R.lim = { pl.parse_mandatory<unsigned long>("lim0"),
              pl.parse_mandatory<unsigned long>("lim1") };
    R.lpb = { pl.parse_mandatory<unsigned int>("lpb0"),
              pl.parse_mandatory<unsigned int>("lpb1") };
    R.mfb = { pl.parse_mandatory<unsigned int>("mfb0"),
              pl.parse_mandatory<unsigned int>("mfb1") };
    auto const stfile = pl.parse_mandatory<std::string>("file-cofact");
    auto const outfile = pl.parse_mandatory<std::string>("out");
    bool const reorder = pl.parse<int>("-reorder");
    double prune = 0;
    pl.parse("prune", prune);
    double C0 = -1;
    pl.parse("C0", C0);
    int radius = 2;
    pl.parse("radius", radius);
    pl.parse("t", R.nthreads);
    std::string cellsfile;
    pl.parse("cells", cellsfile);
    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");
    if (prune > 0 && C0 < 0)
        pl.fail("-prune needs -C0");

    {
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
            if (n[0].bits() <= R.mfb[0] && n[1].bits() <= R.mfb[1])
                R.pairs.emplace_back(std::move(n));
        }
    }

    strategy_file S;
    {
        auto const f = fopen_helper(stfile, "r", true);
        if (!f)
            throw cado::error("cannot read {}", stfile);
        S = facul_strategies::read_file(R.mfb, f.get());
    }

    std::map<sizes_t, cell_stats> stats;

    if (reorder) {
        std::array<std::map<sizes_t, cell_stats>, 6> st;
        for (int o = 1 ; o <= 6 ; o++) {
            strategy_file So;
            for (auto const & [k, chain]: S)
                So[k] = ordering(chain, o);
            st[o - 1] = R(So);
            fmt::print("# ordering {}: {} smooth, {:.3f} s\n", o,
                       total_smooth(st[o - 1]), total_time(st[o - 1]));
        }
        std::array<unsigned long, 7> picked {};
        unsigned long mismatches = 0;
        for (auto & [k, chain]: S) {
            auto const it = st[0].find(k);
            if (it == st[0].end()) {
                picked[0]++;    /* no pairs: left as it was */
                continue;
            }
            int best = 0;
            for (int o = 0 ; o < 6 ; o++) {
                auto const & c = st[o].at(k);
                if (c.smooth != it->second.smooth)
                    mismatches++;
                if (c.time < st[best].at(k).time)
                    best = o;
            }
            chain = ordering(chain, best + 1);
            stats[k] = st[best].at(k);
            picked[best + 1]++;
        }
        /* the pairs of cells that are not in the file */
        for (auto const & [k, c]: st[0])
            if (!S.contains(k))
                stats[k] = c;
        fmt::print("# cells per ordering: {} {} {} {} {} {} (without pairs: {})\n",
                   picked[1], picked[2], picked[3], picked[4], picked[5],
                   picked[6], picked[0]);
        if (mismatches)
            throw cado::error("{} cells found different pairs under different orderings", mismatches);
        fmt::print("# reordered: {} smooth, {:.3f} s\n", total_smooth(stats), total_time(stats));
    } else {
        stats = R(S);
    }

    if (prune > 0) {
        double const T = C0 + total_time(stats);
        double const Y = double(total_smooth(stats));
        unsigned long lost = 0;
        double saved = 0;
        std::vector<sizes_t> drop;
        /* as facul_strategies::accepts_sizes(): a pair of these sizes is
         * found without any method, so the cell stays accepted anyway */
        auto const needs_no_factoring = [&](sizes_t const & k) {
            for (size_t i = 0 ; i < k.size() ; i++)
                if (k[i] > 1 && std::ldexp(1.0, int(k[i]) - 1) >= double(R.lim[i]) * double(R.lim[i]))
                    return false;
            return true;
        };
        for (auto const & [k, chain]: S) {
            auto const s = stats.find(k);
            if (s == stats.end() || s->second.time == 0 || needs_no_factoring(k))
                continue;
            double nsm = 0, nt = 0;
            for (int d0 = -radius ; d0 <= radius ; d0++) {
                for (int d1 = -radius ; d1 <= radius ; d1++) {
                    int const r0 = int(k[0]) + d0, r1 = int(k[1]) + d1;
                    if (r0 < 0 || r1 < 0)
                        continue;
                    auto const n = stats.find({ unsigned(r0), unsigned(r1) });
                    if (n != stats.end()) {
                        nsm += double(n->second.smooth);
                        nt += n->second.time;
                    }
                }
            }
            /* the 0.5 keeps what costs little whatever the sample */
            if (nt * Y > prune * T * (nsm + 0.5)) {
                drop.push_back(k);
                lost += s->second.smooth;
                saved += s->second.time;
            }
        }
        for (auto const & k: drop) {
            S.erase(k);
            stats.erase(k);
        }
        fmt::print("# pruned: dropped {} cells holding {} smooth pairs of {:.0f},"
                   " {:.3f} s of {:.3f}\n", drop.size(), lost, Y, saved, T - C0);
    }

    {
        std::ofstream os(outfile);
        if (!os)
            throw cado::error("cannot write {}", outfile);
        os << fmt::format("# {} refined by refine_strategy_file{}{}\n", stfile,
                reorder ? " -reorder" : "",
                prune > 0 ? fmt::format(" -prune {} -C0 {}", prune, C0) : "");
        facul_strategies::write_file(os, S);
    }

    if (!cellsfile.empty()) {
        std::ofstream os(cellsfile);
        os << "# r0 r1 calls smooth time\n";
        for (auto const & [k, c]: stats)
            os << fmt::format("{} {} {} {} {:.6f}\n", k[0], k[1], c.calls, c.smooth, c.time);
    }

    return EXIT_SUCCESS;
}
