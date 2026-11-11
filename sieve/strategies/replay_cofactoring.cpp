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
#include "facul_strategies.hpp"
#include "macros.h"
#include "params.hpp"
#include "utils_cxx.hpp"

/* Replay the cofactorization step of las on pairs of cofactors that las
 * wrote with -dump-cofactors, under the default strategy or under a
 * strategy file, and say how many relations it finds and how much cpu
 * time it takes, overall and for each pair of cofactor sizes.
 */

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header("Replay the cofactorization step on the pairs of cofactors\n"
                            "that las -dump-cofactors wrote.\n");
    pl.declare_usage("in", "file of pairs, one \"a b N0 N1\" per line");
    pl.declare_usage("lim0", "factor base bound on side 0");
    pl.declare_usage("lim1", "factor base bound on side 1");
    pl.declare_usage("lpb0", "large prime bound on side 0, in bits");
    pl.declare_usage("lpb1", "large prime bound on side 1, in bits");
    pl.declare_usage("mfb0", "cofactor bound on side 0, in bits; larger pairs are skipped");
    pl.declare_usage("mfb1", "cofactor bound on side 1, in bits; larger pairs are skipped");
    pl.declare_usage("ncurves0", "default strategy: number of curves on side 0");
    pl.declare_usage("ncurves1", "default strategy: number of curves on side 1");
    pl.declare_usage("file-cofact", "strategy file to use instead of the default strategy");
    pl.declare_usage("t", "number of threads (default 1)");
    pl.declare_usage("cells", "write per-(r0,r1) statistics to this file");
    pl.declare_usage("smooth", "write the indices (0-based) of the pairs found smooth to this file");
}

namespace {

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

    auto const infile = pl.parse_mandatory<std::string>("in");
    std::vector<unsigned long> lim {
        pl.parse_mandatory<unsigned long>("lim0"),
        pl.parse_mandatory<unsigned long>("lim1") };
    std::vector<unsigned int> lpb {
        pl.parse_mandatory<unsigned int>("lpb0"),
        pl.parse_mandatory<unsigned int>("lpb1") };
    std::vector<unsigned int> mfb {
        pl.parse_mandatory<unsigned int>("mfb0"),
        pl.parse_mandatory<unsigned int>("mfb1") };
    std::vector<int> ncurves { -1, -1 };
    pl.parse("ncurves0", ncurves[0]);
    pl.parse("ncurves1", ncurves[1]);
    std::string stfile;
    pl.parse("file-cofact", stfile);
    unsigned int nthreads = 1;
    pl.parse("t", nthreads);
    std::string cellsfile;
    pl.parse("cells", cellsfile);
    std::string smoothfile;
    pl.parse("smooth", smoothfile);

    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    std::vector<std::vector<cxx_mpz>> pairs;
    std::vector<size_t> pair_index;
    unsigned long skipped = 0;
    {
        std::ifstream is(infile);
        if (!is)
            throw cado::error("cannot read {}", infile);
        std::string line;
        size_t idx = 0;
        while (std::getline(is, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream ls(line);
            int64_t a;
            uint64_t b;
            std::vector<cxx_mpz> n(2);
            if (!(ls >> a >> b >> n[0] >> n[1]))
                throw cado::error("cannot parse line: {}", line);
            if (n[0].bits() > mfb[0] || n[1].bits() > mfb[1]) {
                skipped++;
            } else {
                pairs.emplace_back(std::move(n));
                pair_index.push_back(idx);
            }
            idx++;
        }
    }

    /* las means to pass "perfectly sieved" (factors below lim^2 are
     * prime) when there are no sublattices, which is what we do here.
     * Since sublat_bound defaults to 1, its test for that is never true,
     * so las currently runs without the shortcut; it costs ~0.1% of the
     * cofactorization time. */
    std::unique_ptr<facul_strategies> strat;
    if (stfile.empty()) {
        strat = std::make_unique<facul_strategies>(lim, lpb, mfb, ncurves, true, 0);
    } else {
        auto const f = fopen_helper(stfile, "r", true);
        if (!f)
            throw cado::error("cannot read {}", stfile);
        strat = std::make_unique<facul_strategies>(lim, lpb, mfb, true, f.get(), 0);
    }

    size_t const ncells = (mfb[0] + 1) * (mfb[1] + 1);
    std::vector<std::vector<cell_stats>> per_thread(nthreads, std::vector<cell_stats>(ncells));
    std::vector<std::vector<size_t>> smooth_found(nthreads);

    auto worker = [&](unsigned int t) {
        auto & cells = per_thread[t];
        for (size_t i = t; i < pairs.size(); i += nthreads) {
            auto const & n = pairs[i];
            double const t0 = thread_cputime();
            auto res = facul_all(n, *strat);
            double const t1 = thread_cputime();
            bool ok = true;
            for (auto const & r: res)
                ok = ok && r.status == FACUL_SMOOTH;
            auto & c = cells[n[0].bits() * (mfb[1] + 1) + n[1].bits()];
            c.calls++;
            c.smooth += ok;
            c.time += t1 - t0;
            if (ok)
                smooth_found[t].push_back(pair_index[i]);
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < nthreads; t++)
        threads.emplace_back(worker, t);
    for (auto & th: threads)
        th.join();

    std::vector<cell_stats> total(ncells);
    cell_stats all;
    for (auto const & cells: per_thread) {
        for (size_t k = 0; k < ncells; k++) {
            total[k].calls += cells[k].calls;
            total[k].smooth += cells[k].smooth;
            total[k].time += cells[k].time;
        }
    }
    for (auto const & c: total) {
        all.calls += c.calls;
        all.smooth += c.smooth;
        all.time += c.time;
    }

    fmt::print("# pairs: {} (skipped {} above mfb)\n", all.calls, skipped);
    fmt::print("# smooth: {}\n", all.smooth);
    fmt::print("# cpu time: {:.3f} s, {:.3f} us per pair, {:.3f} ms per relation\n",
               all.time, 1e6 * all.time / double(all.calls),
               all.smooth ? 1e3 * all.time / double(all.smooth) : 0);

    if (!cellsfile.empty()) {
        std::ofstream os(cellsfile);
        os << "# r0 r1 calls smooth time\n";
        for (unsigned int r0 = 0; r0 <= mfb[0]; r0++) {
            for (unsigned int r1 = 0; r1 <= mfb[1]; r1++) {
                auto const & c = total[r0 * (mfb[1] + 1) + r1];
                if (c.calls)
                    os << fmt::format("{} {} {} {} {:.6f}\n", r0, r1,
                                      c.calls, c.smooth, c.time);
            }
        }
    }

    if (!smoothfile.empty()) {
        std::vector<size_t> all_smooth;
        for (auto const & v: smooth_found)
            all_smooth.insert(all_smooth.end(), v.begin(), v.end());
        std::ranges::sort(all_smooth);
        std::ofstream os(smoothfile);
        for (auto i: all_smooth)
            os << i << "\n";
    }

    return EXIT_SUCCESS;
}
