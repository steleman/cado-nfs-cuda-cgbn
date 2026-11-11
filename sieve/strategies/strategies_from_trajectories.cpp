#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"

#include "cado_main.hpp"
#include "facul_method.hpp"
#include "facul_strategies.hpp"
#include "macros.h"
#include "params.hpp"
#include "utils_cxx.hpp"

/* Choose, for each pair of cofactor sizes, how many methods of the
 * default chain to run on each side and in which order, from what
 * cofactoring_trajectories recorded on a sample of pairs, and write the
 * result as a strategy file (las -file-cofact).
 *
 * This is Kleinjung's rule ("Cofactorisation strategies for the number
 * field sieve", 2006): with Y relations for a total cpu time T, a
 * strategy is worth its extra relations if they cost less than T/Y each.
 * For each cell, the candidates are the first K0 methods on side 0 and
 * the first K1 on side 1, along the cheapest interleaving, or nothing;
 * the cell takes the point of the (cost, relations) hull of slope Y/T,
 * and Y/T is found by fixed-point iteration. T is C0, the cpu time it
 * took to produce the pairs, plus the cost of the chosen strategies.
 *
 * The two cofactors of a pair are taken as independent given their
 * sizes: the profile of a side and a size is learnt from every pair that
 * has a cofactor of that size on that side, and a cell only brings its
 * number of pairs.
 *
 * Which pairs a choice of (K0, K1) finds does not depend on the
 * interleaving, and the trajectories say it exactly: -predict writes
 * them, and replay_cofactoring with the strategy file must find those.
 */

static void declare_usage(cxx_param_list & pl)
{
    pl.declare_usage_header("Choose cofactorization strategies, per pair of cofactor sizes,\n"
                            "from the output of cofactoring_trajectories.\n");
    pl.declare_usage("traj", "trajectories (cofactoring_trajectories -out)");
    pl.declare_usage("times", "mean method times (cofactoring_trajectories -times)");
    pl.declare_usage("C0", "cpu seconds that it took to produce the pairs (sieving, trial division)");
    pl.declare_usage("out", "strategy file to write");
    pl.declare_usage("s", "use this many relations per cpu second instead of finding it");
    pl.declare_usage("grid", "chain lengths to try, comma-separated (default 1..14, then up to 43)");
    pl.declare_usage("counts", "the pairs per cell, as las -stats-cofact writes them (\"r0 r1 calls successes\"), instead of the pairs of -traj; for trajectories of synthetic cofactors");
    pl.declare_usage("cells", "write \"r0 r1 K0 K1 pairs relations cost path\" for each accepted cell");
    pl.declare_usage("predict", "write the indices (0-based) of the pairs of -traj that the chosen strategies find");
}

namespace {

/* the longest chain we consider */
constexpr int KM = 64;

struct side_outcome {
    unsigned int j = 0; /* method that first splits, 1-based; 0 if none */
    char o = 'U';       /* T (nothing to do), N, S, A, U */
    unsigned int e = 0; /* for A: method by which the composites are done */
    char x = '-';       /* for A: S or N */
    std::vector<double> aux;    /* cumulative aux times, seconds */

    /* whether the first K methods of the chain make this side smooth */
    bool smooth_within(int K) const {
        switch (o) {
            case 'T': return true;
            case 'S': return int(j) <= K;
            case 'A': return x == 'S' && int(e) <= K;
            default: return false;
        }
    }
};

struct trajectory {
    unsigned int r[2];
    side_outcome side[2];
};

std::vector<trajectory> read_trajectories(std::string const & filename)
{
    std::ifstream is(filename);
    if (!is)
        throw cado::error("cannot read {}", filename);
    std::vector<trajectory> res;
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ls(line);
        trajectory t;
        ls >> t.r[0] >> t.r[1];
        for (auto & s: t.side) {
            ls >> s.j >> s.o >> s.e >> s.x;
            std::string tok;
            while (ls >> tok && tok != ";")
                s.aux.push_back(1e-9 * std::stod(tok));
        }
        if (!ls.eof() || ls.bad())
            throw cado::error("cannot parse line: {}", line);
        res.push_back(std::move(t));
    }
    return res;
}

/* T[s][r][k]: mean time in seconds of method k (1-based) on side s for
 * a cofactor of r bits */
using times_table = std::vector<std::vector<std::vector<double>>>;

times_table read_times(std::string const & filename)
{
    std::ifstream is(filename);
    if (!is)
        throw cado::error("cannot read {}", filename);
    times_table T(2);
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ls(line);
        unsigned int s, r, k;
        double t;
        unsigned long n;
        if (!(ls >> s >> r >> k >> t >> n) || s > 1)
            throw cado::error("cannot parse line: {}", line);
        if (T[s].size() <= r)
            T[s].resize(r + 1);
        if (T[s][r].size() <= k)
            T[s][r].resize(k + 1, 0);
        T[s][r][k] = 1e-9 * t;
    }
    return T;
}

/* What the pairs that have a cofactor of a given size on a given side
 * say about it, as fractions of these pairs; index i is "within the
 * first i methods". */
struct profile {
    double triv = 0;                /* nothing to do */
    double unsplit[KM + 2] = {};    /* something to do, not split yet */
    double alive_split[KM + 2] = {};/* split, and not shown non-smooth */
    double done[KM + 2] = {};       /* nothing to do, or fully factored */
    double aux[KM + 2] = {};        /* mean cost of the work after a split */
};

std::vector<profile> make_profiles(std::vector<trajectory> const & traj,
        int s, int & Kmax)
{
    struct counts {
        double n = 0, triv = 0;
        double split_alive[KM + 2] = {}, split_dead[KM + 2] = {};
        double done[KM + 2] = {}, aux[KM + 2] = {};
    };
    std::vector<counts> C;
    for (auto const & t: traj) {
        auto const & o = t.side[s];
        auto const r = t.r[s];
        if (C.size() <= r)
            C.resize(r + 1);
        auto & P = C[r];
        P.n++;
        Kmax = std::max(Kmax, int(std::max(o.j, o.e)));
        if (o.j > KM || o.e > KM)
            throw cado::error("chains longer than {} methods are not supported", KM);
        switch (o.o) {
            case 'T': P.triv++; break;
            case 'U': break;
            case 'N': P.split_dead[o.j]++; break;
            case 'S': P.split_alive[o.j]++; P.done[o.j]++; break;
            case 'A':
                P.split_alive[o.j]++;
                if (o.x == 'S')
                    P.done[o.e]++;
                /* a chain of K methods spends on the aux steps j+1..K,
                 * or up to e if they stop there */
                for (int K = int(o.j) + 1 ; K <= KM ; K++) {
                    int last = K;
                    if (o.e && int(o.e) < last)
                        last = int(o.e);
                    int const nsteps = last - int(o.j);
                    if (nsteps > 0 && !o.aux.empty())
                        P.aux[K] += o.aux[std::min(size_t(nsteps), o.aux.size()) - 1];
                }
                break;
            default:
                throw cado::error("unknown outcome {}", o.o);
        }
    }
    std::vector<profile> res(std::max(C.size(), size_t(2)));
    for (size_t r = 0 ; r < res.size() ; r++) {
        auto & c = res[r];
        if (r >= C.size() || C[r].n == 0) {
            /* no data: a cofactor of at most one bit is 1 */
            if (r <= 1) {
                c.triv = 1;
                std::fill(std::begin(c.done), std::end(c.done), 1);
            }
            continue;
        }
        auto const & P = C[r];
        c.triv = P.triv / P.n;
        double sa = 0, sd = 0, dn = 0;
        for (int i = 0 ; i <= KM + 1 ; i++) {
            if (i) {
                sa += P.split_alive[i];
                sd += P.split_dead[i];
                dn += P.done[i];
            }
            c.alive_split[i] = sa / P.n;
            c.unsplit[i] = (P.n - P.triv - sa - sd) / P.n;
            c.done[i] = (P.triv + dn) / P.n;
            c.aux[i] = P.aux[i] / P.n;
        }
    }
    return res;
}

struct candidate {
    int K0 = -1, K1 = -1;   /* -1: the cell is rejected */
    double succ = 0, cost = 0;
    std::string path;       /* the sides of the methods, in order */
};

/* The candidates of one cell that are on the upper convex hull of
 * (cost, relations) per pair, by increasing cost. */
std::vector<candidate> cell_hull(profile const & c0,
        profile const & c1, std::vector<double> const & T0,
        std::vector<double> const & T1, std::vector<int> const & g0,
        std::vector<int> const & g1)
{
    std::vector<candidate> cands(1);
    for (int const K0: g0) {
        for (int const K1: g1) {
            auto alive0 = [&](int i) { return c0.triv + c0.alive_split[i] + (i < K0 ? c0.unsplit[i] : 0); };
            auto alive1 = [&](int j) { return c1.triv + c1.alive_split[j] + (j < K1 ? c1.unsplit[j] : 0); };
            /* V(i, j): the expected cost of what remains once i methods
             * of side 0 and j of side 1 have run; a method only runs if
             * its side is still unsplit and the other side alive */
            int const w = K1 + 1;
            std::vector<double> V((K0 + 1) * w, 0);
            std::vector<char> D((K0 + 1) * w, 0);
            for (int i = K0 ; i >= 0 ; i--) {
                for (int j = K1 ; j >= 0 ; j--) {
                    if (i == K0 && j == K1)
                        continue;
                    double best = INFINITY;
                    char d = 0;
                    if (i < K0) {
                        double const x = c0.unsplit[i] * alive1(j) * T0[i + 1] + V[(i + 1) * w + j];
                        if (x < best) { best = x; d = '0'; }
                    }
                    if (j < K1) {
                        double const x = c1.unsplit[j] * alive0(i) * T1[j + 1] + V[i * w + j + 1];
                        if (x < best) { best = x; d = '1'; }
                    }
                    V[i * w + j] = best;
                    D[i * w + j] = d;
                }
            }
            candidate c;
            c.K0 = K0;
            c.K1 = K1;
            for (int i = 0, j = 0 ; i < K0 || j < K1 ; ) {
                char const d = D[i * w + j];
                c.path.push_back(d);
                if (d == '0') i++; else j++;
            }
            c.succ = c0.done[K0] * c1.done[K1];
            c.cost = V[0] + c0.aux[K0] * alive1(K1) + c1.aux[K1] * alive0(K0);
            cands.push_back(std::move(c));
        }
    }
    std::ranges::sort(cands, [](candidate const & a, candidate const & b) {
            return a.cost < b.cost || (a.cost == b.cost && a.succ > b.succ);
            });
    std::vector<candidate> hull;
    for (auto & c: cands) {
        if (!hull.empty() && c.succ <= hull.back().succ)
            continue;
        while (hull.size() >= 2) {
            auto const & p = hull[hull.size() - 2];
            auto const & q = hull.back();
            double const cross = (q.cost - p.cost) * (c.succ - p.succ) - (q.succ - p.succ) * (c.cost - p.cost);
            if (cross < 0)
                break;
            hull.pop_back();
        }
        hull.push_back(std::move(c));
    }
    return hull;
}

/* the first K methods of the default chain */
std::vector<facul_method::parameters> default_prefix(int K)
{
    auto chain = facul_strategy_oneside::default_strategy(std::max(K - 3, 0));
    ASSERT_ALWAYS(int(chain.size()) >= K);
    chain.resize(K);
    return chain;
}

std::vector<int> parse_grid(std::string const & s)
{
    std::vector<int> res;
    std::istringstream is(s);
    std::string tok;
    while (std::getline(is, tok, ','))
        res.push_back(std::stoi(tok));
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

    auto const trajfile = pl.parse_mandatory<std::string>("traj");
    auto const timesfile = pl.parse_mandatory<std::string>("times");
    auto const C0 = pl.parse_mandatory<double>("C0");
    auto const outfile = pl.parse_mandatory<std::string>("out");
    double fixed_s = 0;
    pl.parse("s", fixed_s);
    std::string gridstr = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,22,24,26,28,30,33,36,40,43";
    pl.parse("grid", gridstr);
    std::string countsfile;
    pl.parse("counts", countsfile);
    std::string cellsfile;
    pl.parse("cells", cellsfile);
    std::string predictfile;
    pl.parse("predict", predictfile);
    if (pl.warn_unused())
        pl.fail("Not all arguments were parsed");

    auto const grid = parse_grid(gridstr);
    auto const traj = read_trajectories(trajfile);
    auto const T = read_times(timesfile);

    int Kmax = 0;
    std::vector<profile> const prof[2] = {
        make_profiles(traj, 0, Kmax),
        make_profiles(traj, 1, Kmax),
    };

    /* The choice in a cell does not depend on how many pairs it has, only
     * on the profiles of its two sizes: so every pair of sizes seen on
     * their sides gets one, also those that the sample has no pair of.
     * The pairs of the sample weigh the cells in the search for s. */
    std::map<std::pair<unsigned int, unsigned int>, double> cells;
    std::set<unsigned int> seen[2];
    for (auto const & t: traj) {
        seen[0].insert(t.r[0]);
        seen[1].insert(t.r[1]);
    }
    for (auto const r0: seen[0])
        for (auto const r1: seen[1])
            cells[{r0, r1}] = 0;
    if (countsfile.empty()) {
        for (auto const & t: traj)
            cells[{t.r[0], t.r[1]}]++;
    } else {
        std::ifstream is(countsfile);
        if (!is)
            throw cado::error("cannot read {}", countsfile);
        unsigned int r0, r1;
        unsigned long calls, successes;
        while (is >> r0 >> r1 >> calls >> successes)
            if (calls && seen[0].contains(r0) && seen[1].contains(r1))
                cells[{r0, r1}] += double(calls);
    }

    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<candidate>>> hulls;
    for (auto const & kv: cells) {
        auto const & r = kv.first;
        unsigned int const rr[2] = { r.first, r.second };
        std::vector<int> g[2];
        std::vector<double> const * Ts[2];
        static std::vector<double> const none;
        for (int s = 0 ; s < 2 ; s++) {
            auto const & p = prof[s][rr[s]];
            Ts[s] = rr[s] < T[s].size() ? &T[s][rr[s]] : &none;
            if (p.triv >= 1.0 - 1e-12) {
                g[s] = { 0 };
                continue;
            }
            for (int const K: grid)
                if (K >= 1 && K <= Kmax && K < int(Ts[s]->size()))
                    g[s].push_back(K);
        }
        hulls.emplace_back(r, cell_hull(prof[0][rr[0]], prof[1][rr[1]],
                    *Ts[0], *Ts[1], g[0], g[1]));
    }

    /* For a slope s, each cell takes the last point of its hull that is
     * reached by segments of slope at least s */
    auto evaluate = [&](double s, double & Y, double & Tt, std::vector<size_t> * choice) {
        Y = 0;
        Tt = C0;
        for (auto const & [r, H]: hulls) {
            size_t k = 0;
            while (k + 1 < H.size() && (H[k + 1].succ - H[k].succ) >= s * (H[k + 1].cost - H[k].cost))
                k++;
            double const n = cells.at(r);
            Y += n * H[k].succ;
            Tt += n * H[k].cost;
            if (choice)
                choice->push_back(k);
        }
    };
    double Y, Tt;
    double s = fixed_s;
    if (s <= 0) {
        s = 0.1;
        for (int it = 0 ; it < 100 ; it++) {
            evaluate(s, Y, Tt, nullptr);
            double const ns = Y / Tt;
            if (std::fabs(ns - s) < 1e-10 * s)
                break;
            s = ns;
        }
    }
    std::vector<size_t> choice;
    evaluate(s, Y, Tt, &choice);
    fmt::print("# s = {:.6g}, {:.1f} relations expected in {:.1f} cpu seconds, {:.6g} per second\n",
            s, Y, Tt, Y / Tt);

    std::map<std::pair<unsigned int, unsigned int>, candidate const *> chosen;
    {
        std::ofstream os(outfile);
        if (!os)
            throw cado::error("cannot write {}", outfile);
        os << fmt::format("# from {} by strategies_from_trajectories, s = {:.6g}\n", trajfile, s);
        std::ofstream cs;
        facul_strategies::strategy_file S;
        if (!cellsfile.empty())
            cs.open(cellsfile);
        for (size_t i = 0 ; i < hulls.size() ; i++) {
            auto const & [r, H] = hulls[i];
            auto const & c = H[choice[i]];
            if (c.K0 < 0)
                continue;
            chosen[r] = &c;
            double const n = cells[r];
            if (cs.is_open())
                cs << fmt::format("{} {} {} {} {} {:.2f} {:.6f} {}\n",
                        r.first, r.second, c.K0, c.K1, n, n * c.succ, n * c.cost,
                        c.path.empty() ? "-" : c.path);
            /* nothing to run: the cell is accepted all the same when
             * it is not in the file */
            if (c.path.empty())
                continue;
            std::vector<facul_method::parameters> const pre[2] = {
                default_prefix(c.K0), default_prefix(c.K1) };
            std::vector<facul_method::parameters_with_side> chain;
            size_t pos[2] = { 0, 0 };
            for (char const d: c.path) {
                int const side = d - '0';
                facul_method::parameters_with_side m;
                static_cast<facul_method::parameters &>(m) = pre[side][pos[side]++];
                m.side = side;
                chain.push_back(m);
            }
            S[{ r.first, r.second }] = chain;
        }
        facul_strategies::write_file(os, S);
    }

    if (!predictfile.empty()) {
        std::ofstream os(predictfile);
        for (size_t i = 0 ; i < traj.size() ; i++) {
            auto const & t = traj[i];
            auto it = chosen.find({t.r[0], t.r[1]});
            bool ok;
            if (it == chosen.end())
                /* a rejected cell still yields the pairs that need no
                 * factoring at all */
                ok = t.side[0].o == 'T' && t.side[1].o == 'T';
            else
                ok = t.side[0].smooth_within(it->second->K0)
                    && t.side[1].smooth_within(it->second->K1);
            if (ok)
                os << i << "\n";
        }
    }

    return EXIT_SUCCESS;
}
