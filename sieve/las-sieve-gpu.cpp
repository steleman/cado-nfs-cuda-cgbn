#include "cado.h" // IWYU pragma: keep

/* GPU siever (las-sieve-gpu-kernels.cu): host side.
 *
 * las_sieve_gpu_run() gathers what the GPU needs for one special-q --
 * the lognorm data of both sides (wss.lognorms), the bucket-sieved
 * factor base entries (all parts above the small sieve) and the small
 * sieve entries that small_sieve_init made (wss.ssd) -- and sieves the
 * whole region on the GPU.
 *
 * It is used for every special-q it applies to (two sides with factor
 * bases, no sublattices, logI <= 16), unless CADO_LAS_GPU_SIEVE=0 (atoi).
 *
 * With CADO_LAS_GPU_SIEVE_CHECK=1, las still sieves on the CPU, and
 * process_bucket_region compares its sieve arrays, after SminusS, with
 * the GPU's at every position where gcd(i,j) = 1 (the only ones that can
 * survive); las_sieve_gpu_report() prints the counts.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <gmp.h>

#include "fb.hpp"
#include "cxx_mpz.hpp"
#include "mpz_poly.h"
#include "las-config.hpp"
#include "las-norms.hpp"
#include "las-qlattice.hpp"
#include "las-sieve-gpu.hpp"
#include "las-smallsieve.hpp"
#include "las-threads-work-data.hpp"
#include "trialdiv.hpp"
#include "macros.h"
#include "timing.h"
#include "verbose.hpp"

#if defined(HAVE_CUDA) && !defined(SIQS_SIEVE) && !defined(SUPPORT_LARGE_Q) && !defined(TRACE_K)
#include "las-sieve-gpu-kernels.h"

namespace {

bool check_mode()
{
    static int const e = []() {
        const char * s = getenv("CADO_LAS_GPU_SIEVE_CHECK");
        return s ? atoi(s) : 0;
    }();
    return e != 0;
}

lsg_ctx * context()
{
    static thread_local lsg_ctx * ctx = nullptr;
    static thread_local bool tried = false;
    if (!tried) {
        tried = true;
        ctx = lsg_ctx_create();
    }
    return ctx;
}

/* the bucket-sieved entries of a side, with their logp; odd q go to the
 * GPU as they are, powers of 2 are transformed on the host per q */
struct side_table {
    std::vector<lsg_lat> lats;
    lsg_table * dev = nullptr;      /* lats, on the GPU */
    ~side_table() { lsg_table_free(dev); }
    std::vector<slice_index_t> slice_index;     /* local -> global */
    struct pow2 { fb_entry_general const * e; uint16_t hint, slice; uint8_t logp; };
    std::vector<pow2> pow2s;
};

std::mutex tables_lock;
std::map<std::pair<void const *, int>, std::shared_ptr<side_table>> tables;

std::shared_ptr<side_table> get_table(nfs_work const & ws, int side)
{
    nfs_work::side_data const & wss(ws.sides[side]);
    std::lock_guard<std::mutex> const dummy(tables_lock);
    auto key = std::make_pair((void const *) wss.fbs, side);
    auto it = tables.find(key);
    if (it != tables.end()) return it->second;
    auto T = std::make_shared<side_table>();
    uint32_t ls = 0;
    for (int level = 1; level <= wss.fbs->get_toplevel(); level++) {
        fb_factorbase::slicing::part const & P = wss.fbs->get_part(level);
        P.foreach_slice([&](auto const & s) {
            using E = std::decay_t<decltype(s)>::entry_t;
            uint8_t const logp = s.get_logp();
            T->slice_index.push_back(s.get_index());
            slice_offset_t i_entry = 0;
            for (auto const & e : s) {
                if constexpr (E::is_general_type) {
                    if (!(e.q & 1)) {
                        T->pow2s.push_back({ &e, i_entry, (uint16_t) ls, logp });
                        i_entry++;
                        continue;
                    }
#ifndef BUCKET_SIEVE_POWERS
                    if (e.k > 1) { i_entry++; continue; }
#endif
                }
                for (int i = 0; i < e.get_nr_roots(); i++) {
                    lsg_lat L {};
                    L.L.q = e.get_q();
                    L.L.p = e.p;
                    L.L.r = e.get_r(i);
                    L.L.hint = i_entry;
                    L.L.slice = (uint16_t) ls;
                    L.L.proj = e.get_proj(i);
                    L.logp = logp;
                    T->lats.push_back(L);
                }
                i_entry++;
            }
            ls++;
        });
    }
    T->dev = lsg_table_create(T->lats.data(), T->lats.size());
    tables[key] = T;
    return T;
}

/* check mode: the GPU's arrays of the special-q being sieved, per
 * nfs_work */
struct check_data {
    std::vector<uint8_t> S[2];
    std::vector<lsg_small> small[2];
    int logI = 0;
};
std::mutex check_lock;
std::map<nfs_work const *, std::shared_ptr<check_data>> checks;
std::atomic<unsigned long> n_compared(0), n_mismatch(0), n_regions(0);
std::mutex stats_lock;
double stats_wall = 0;
unsigned long stats_calls = 0;
size_t stats_survivors = 0;
size_t stats_cof_rejected = 0;    /* by the GPU cofactor filter */

} /* namespace */

namespace {

/* on by default (where it applies); CADO_LAS_GPU_SIEVE=0 for the CPU */
bool real_mode()
{
    static int const e = []() {
        const char * s = getenv("CADO_LAS_GPU_SIEVE");
        return s ? atoi(s) : 1;
    }();
    return e != 0;
}

bool applicable(nfs_work const & ws, qlattice_basis const & Q)
{
    int const nsides = ws.sides.size();
    if (nsides != 2 || Q.sublat.m > 1 || ws.conf.logI > 16) return false;
    if (ws.conf.logI > LOG_BUCKET_REGION) return false;
    if (!ws.conf.needs_resieving()) return false;
    for (int side = 0; side < nsides; side++)
        if (ws.sides[side].no_fb()) return false;
    return true;
}

/* the inputs of lsg_sieve for this special-q */
struct gpu_inputs {
    lsg_qinfo qi {};
    lsg_side sd[2] {};
    std::vector<double> Gend[2], Geq[2];
    std::vector<lsg_lat> lats[2];
    std::vector<lsg_small> small[2];
    std::shared_ptr<side_table> T[2];
    uint8_t bound[2] = { 0, 0 };
};

bool build_inputs(nfs_work & ws, qlattice_basis const & Q, gpu_inputs & in)
{
    int const nsides = 2;
    lsg_qinfo & qi(in.qi);
    lsg_side * sd = in.sd;
    auto & Gend = in.Gend;
    auto & Geq = in.Geq;
    auto & lats = in.lats;
    auto & small = in.small;

    qi.a0 = Q.a0; qi.b0 = Q.b0; qi.a1 = Q.a1; qi.b1 = Q.b1;
    if (Q.doing.is_prime()) {
        qi.qfac[qi.nqfac++] = mpz_get_ui(Q.doing.p);
    } else {
        for (auto f : Q.doing.prime_factors) {
            if (qi.nqfac == 8) return false;
            qi.qfac[qi.nqfac++] = f;
        }
    }
    qi.logI = ws.conf.logI;
    qi.J = ws.J;

    for (int side = 0; side < nsides; side++) {
        nfs_work::side_data & wss(ws.sides[side]);
        lognorm_smart const & L = wss.lognorms;
        lsg_norm & N = sd[side].norm;
        N.degree = L.fijd.degree();
        /* fijd[k] is a proxy: convert it to double explicitly (in a
         * conditional expression with an int it would become an int) */
        N.u0 = (double) L.fijd[0];
        N.u1 = N.degree >= 1 ? (double) L.fijd[1] : 0.0;
        double const modscale = L.scale / 0x100000;
        N.modscale = modscale;
        N.offset = 0x3FF00000 - LOGNORM_GUARD_BITS / modscale;
        {
            /* LOGNORM_COMMON_HANDLE_ORIGIN, with sl.m == 1 */
            double const norm = (log2(fabs(L.fijd.lc())) - L.fijd.degree() * log2(1)) * L.scale;
            N.origin = LOGNORM_GUARD_BITS + (unsigned char) (norm);
        }
        for (int k = 0; k < 257; k++) N.cexp2[k] = L.cexp2[k];
        if (N.degree > 1) {
            Gend[side].assign(L.G.endpoints.begin(), L.G.endpoints.end());
            for (auto const & e : L.G.equations) {
                Geq[side].push_back(e.first);
                Geq[side].push_back(e.second);
            }
            N.nseg = L.G.equations.size();
            ASSERT_ALWAYS(Gend[side].size() == N.nseg + 1);
        }
        sd[side].G_endpoints = Gend[side].data();
        sd[side].G_equations = Geq[side].data();

        auto T = get_table(ws, side);
        in.T[side] = T;
        sd[side].table = T->dev;
        lats[side].clear();     /* the table is on the GPU; only the powers of 2 here */
        for (auto const & pw : T->pow2s) {
            fb_entry_general const & e = *pw.e;
            if (!Q.is_coprime_to(e.p)) continue;
            fb_entry_general tr;
            e.transform_roots(tr, Q);
            for (unsigned char i = 0; i != tr.nr_roots; i++) {
                lsg_lat X {};
                X.L.q = tr.get_q();
                X.L.p = e.p;
                X.L.r = tr.get_r(i);
                X.L.hint = pw.hint;
                X.L.slice = pw.slice;
                X.L.proj = tr.get_proj(i);
                X.logp = pw.logp;
                X.transformed = 1;
                lats[side].push_back(X);
            }
        }
        sd[side].lats = lats[side].data();
        sd[side].nlats = lats[side].size();

        auto const * ssd = dynamic_cast<las_small_sieve_data const *>(wss.ssd.get());
        if (!ssd) return false;
        {
            /* search_survivors' bound */
            unsigned int const b = wss.lognorms.bound + wss.ssd->constant_logp;
            in.bound[side] = (unsigned char) (b < 254 ? b : 254);
        }
        size_t const nres = ssd->get_resieve_end_offset();
        size_t k = 0;
        for (auto const & s : ssd->get_ssps()) {
            lsg_small x {};
            x.q = s.get_p();
            x.r = s.get_r();
            x.logp = s.logp;
            x.kind = 0;
            x.resieve = k++ < nres;
            small[side].push_back(x);
        }
        for (auto const & s : ssd->get_ssp()) {
            if (s.is_sublat_rowwise()) return false;
            lsg_small x {};
            x.logp = s.logp;
            if (s.is_proj()) {
                x.kind = 1;
                x.q = s.get_q();
                x.g = s.get_g();
                x.r = s.get_U();
                /* as resieve_small_bucket_region: projective entries
                 * that sieve whole lines, not powers */
                x.resieve = !s.is_pow() && !s.is_pow2() && x.q == 1
                    && x.g >= wss.fbK.td_thresh;
            } else {
                x.kind = 0;
                x.q = s.raw_p();
                x.r = s.raw_r();
            }
            small[side].push_back(x);
        }
        if (getenv("CADO_LAS_GPU_SIEVE_DEBUG")) {
            static std::atomic<int> printed(0);
            if (printed++ < 2)
                for (auto const & s : ssd->get_ssp())
                    fprintf(stderr, "# side %d ssp p=%u r=%u off=%u logp=%u proj=%d pow2=%d pat=%d pow=%d\n",
                            side, s.raw_p(), s.raw_r(), s.raw_offset(), s.logp,
                            s.is_proj(), s.is_pow2(), s.is_pattern_sieved(), s.is_pow());
        }
        /* by q: the kernel gives warps to the small ones (the order does
         * not matter otherwise) */
        std::stable_sort(small[side].begin(), small[side].end(),
                [](lsg_small const & a, lsg_small const & b) { return a.q < b.q; });
        sd[side].small = small[side].data();
        sd[side].nsmall = small[side].size();
    }
    return true;
}

} /* namespace */

bool las_sieve_gpu_enabled(nfs_work const & ws, qlattice_basis const & Q)
{
    return real_mode() && !check_mode() && applicable(ws, Q) && context();
}

/* filter_survivors_by_region as one table of the pair (S0, S1): bit
 * s0 * 256 + s1 is A[est0[s0] * 256 + est1[s1]], with est as there.
 * Rebuilt when what it depends on changes (the strategies, the bounds of
 * the siever config that accept_estimate comes from, the scales and the
 * offsets); per thread, as the GPU context is. */
/* true if the variable is set to 0 */
static bool getenv_flag_off(char const * name)
{
    char const * e = getenv(name);
    return e && strcmp(e, "0") == 0;
}

struct accept_cache {
    facul_strategies const * strategies = nullptr;
    unsigned int mfb[2] = { 0, 0 };
    unsigned long lim[2] = { 0, 0 };
    double scale[2] = { 0, 0 };
    int offset[2] = { 0, 0 };
    uint64_t version = 0;
    std::vector<uint32_t> bits;
};

static uint32_t const * accept_table(nfs_work const & ws, nfs_work_cofac const & wc,
        uint64_t & version)
{
    static thread_local accept_cache C;
    double scale[2];
    int offset[2];
    for (int side = 0; side < 2; side++) {
        scale[side] = ws.sides[side].lognorms.scale;
        offset[side] = LOGNORM_GUARD_BITS + ws.sides[side].ssd->constant_logp;
    }
    bool same = C.version && C.strategies == wc.strategies;
    for (int side = 0; same && side < 2; side++)
        same = C.mfb[side] == wc.sc.sides[side].mfb && C.lim[side] == wc.sc.sides[side].lim
            && C.scale[side] == scale[side] && C.offset[side] == offset[side];
    if (!same) {
        std::array<std::array<uint8_t, 256>, 2> est;
        for (int side = 0; side < 2; side++)
            for (int v = 0; v < 256; v++) {
                double const e = (v - offset[side]) / scale[side];
                est[side][v] = e < 0 ? 0 : (e > 255 ? 255 : (uint8_t) e);
            }
        auto const & A = wc.accept_estimate;
        C.bits.assign(65536 / 32, 0);
        for (unsigned int s0 = 0; s0 < 256; s0++)
            for (unsigned int s1 = 0; s1 < 256; s1++)
                if (A[est[0][s0] * 256 + est[1][s1]]) {
                    unsigned int const k = s0 * 256 + s1;
                    C.bits[k >> 5] |= 1u << (k & 31);
                }
        C.strategies = wc.strategies;
        for (int side = 0; side < 2; side++) {
            C.mfb[side] = wc.sc.sides[side].mfb;
            C.lim[side] = wc.sc.sides[side].lim;
            C.scale[side] = scale[side];
            C.offset[side] = offset[side];
        }
        C.version++;
    }
    version = C.version;
    return C.bits.data();
}

/* The parameters of the cofactor filter for this q (see lsg_cofilter):
 * false if the GPU cannot do it (the CPU then sees every survivor). The
 * trial division primes are kept per thread and get a new version when
 * the lists change. */
struct td_cache {
    trialdiv_data const * td[2] = { nullptr, nullptr };
    size_t n[2] = { 0, 0 };
    std::vector<uint32_t> p[2];
    uint64_t version = 0;
};

static bool cofilter_params(nfs_work const & ws, qlattice_basis const & Q,
        lsg_cofilter & P)
{
    static thread_local td_cache T;
    char const * e = getenv("CADO_LAS_GPU_COFILTER");
    if (e && strcmp(e, "0") == 0) return false;
    memset(&P, 0, sizeof(P));
    int const logI = ws.conf.logI;
    unsigned int const jbits = ws.J ? (unsigned int) (64 - __builtin_clzll(ws.J)) : 1;
    unsigned int const xbits = std::max((unsigned int) (logI - 1) + 1, jbits);
    unsigned int need = 0;
    bool td_changed = false;
    for (int side = 0; side < 2; side++) {
        nfs_work::side_data const & wss(ws.sides[side]);
        lsg_cofilter_side & S = P.s[side];
        if (wss.no_fb() || !wss.td) {
            S.deg = -1;
            continue;
        }
        cxx_mpz_poly const & f = wss.lognorms.fij;
        int const d = f->deg;
        if (d < 0 || d > LSG_COF_MAXDEG) return false;
        S.deg = d;
        size_t maxc = 0;
        for (int k = 0; k <= d; k++) {
            mpz_srcptr c = mpz_poly_coeff_const(f, k);
            size_t const b = mpz_sizeinbase(c, 2);
            if (b > 512) return false;
            maxc = std::max(maxc, b);
            S.sign[k] = (int8_t) mpz_sgn(c);
            size_t cnt = 0;
            mpz_export(S.limbs[k], &cnt, -1, sizeof(uint32_t), 0, 0, c);
        }
        /* each term below 2^(maxc + d xbits), and their sums */
        unsigned int const nb = (unsigned int) maxc + (unsigned int) d * xbits + 6;
        need = std::max(need, nb);
        siever_side_config const & sc = ws.conf.sides[side];
        S.lpb = sc.lpb;
        S.mfb = sc.mfb;
        S.lim = sc.lim;
        if (!Q.doing.is_prime() && (int) side == Q.doing.side) {
            for (auto q : Q.doing.prime_factors) {
                if (S.nqf == 8 || q >> 32) return false;
                S.qf[S.nqf++] = (uint32_t) q;
            }
        }
        if (T.td[side] != wss.td || T.n[side] != wss.td->size()) {
            T.td[side] = wss.td;
            T.n[side] = wss.td->size();
            /* in groups for the GPU: count, product (< 2^32), primes */
            T.p[side].clear();
            std::vector<uint32_t> grp;
            uint64_t prod = 1;
            auto flush = [&]() {
                if (grp.empty()) return;
                T.p[side].push_back((uint32_t) grp.size());
                T.p[side].push_back((uint32_t) prod);
                T.p[side].insert(T.p[side].end(), grp.begin(), grp.end());
                grp.clear();
                prod = 1;
            };
            for (auto const & t : *wss.td) {
                if (t.p == 1) continue;
                if (t.p >> 32) return false;
                if (prod * t.p >> 32) flush();
                grp.push_back((uint32_t) t.p);
                prod *= t.p;
            }
            flush();
            td_changed = true;
        }
        S.td = T.p[side].data();
        S.ntd = (uint32_t) T.p[side].size();
    }
    if (td_changed || !T.version) T.version++;
    P.td_version = T.version;
    /* CADO_LAS_GPU_COFILTER_BITS=512 forces the wider kernel (testing) */
    char const * wb = getenv("CADO_LAS_GPU_COFILTER_BITS");
    if (need <= 256 && !(wb && atoi(wb) == 512)) P.bits = 256;
    else if (need <= 512) P.bits = 512;
    else return false;
    return true;
}

bool las_sieve_gpu_sieve(nfs_work & ws, qlattice_basis const & Q,
        std::vector<gpu_sieve_region> & regions, nfs_work_cofac const * wc)
{
    lsg_ctx * ctx = context();
    if (!ctx) return false;
    gpu_inputs in;
    if (!build_inputs(ws, Q, in)) return false;
    lsg_out out {};
    double const t0 = wct_seconds();
    uint64_t accept_version = 0;
    uint32_t const * accept = nullptr;
    if (wc && !wc->accept_estimate.empty() && !getenv_flag_off("CADO_LAS_GPU_FILTER"))
        accept = accept_table(ws, *wc, accept_version);
    lsg_cofilter cof;
    bool const use_cof = cofilter_params(ws, Q, cof);
    lsg_sieve(ctx, &in.qi, in.sd, 2, nullptr, in.bound, &out, accept, accept_version,
            use_cof ? &cof : nullptr);

    /* by region; a region is 2^LOG_BUCKET_REGION consecutive positions */
    int const logB = LOG_BUCKET_REGION;
    uint32_t const mask = (1u << logB) - 1;
    /* the region of each region number (a flat table: a std::map lookup
     * per hit was a tenth of las's CPU time) */
    size_t const nreg = (((size_t) ws.J << ws.conf.logI) >> logB) + 1;
    std::vector<int32_t> idx(nreg, -1);
    regions.clear();
    std::vector<lsg_survivor> sv(out.surv, out.surv + out.nsurv);
    std::sort(sv.begin(), sv.end(), [](lsg_survivor const & a, lsg_survivor const & b) {
            return a.x < b.x; });
    size_t nrejected = 0;
    for (auto const & v : sv) {
        if (v.pad) {        /* check_leftover_norm() would reject it */
            nrejected++;
            continue;
        }
        unsigned int const N = v.x >> logB;
        ASSERT_ALWAYS(N < nreg);
        if (idx[N] < 0) {
            idx[N] = (int32_t) regions.size();
            regions.emplace_back();
            regions.back().N = N;
        }
        gpu_sieve_region & R = regions[idx[N]];
        R.x.push_back(v.x & mask);
        R.S[0].push_back(v.S[0]);
        R.S[1].push_back(v.S[1]);
    }
    /* only the hits at a survivor that is kept: the CPU looks at no other
     * (divide_known_primes), and with the cofactor filter most survivors
     * of a region are gone; R.x is sorted */
    auto const at_survivor = [&](gpu_sieve_region const & R, uint32_t pos) {
        return std::binary_search(R.x.begin(), R.x.end(), (uint16_t) pos);
    };
    for (size_t k = 0; k < out.nlarge; k++) {
        lsg_hit_large const & h = out.large[k];
        size_t const N = h.x >> logB;
        if (N >= nreg || idx[N] < 0) continue;
        if (!at_survivor(regions[idx[N]], h.x & mask)) continue;
        bucket_update_t<1, longhint_t> u(h.x & mask, 0, h.hint,
                in.T[h.side]->slice_index[h.slice]);
        regions[idx[N]].purged[h.side].push_back(u);
    }
    for (size_t k = 0; k < out.nsmall; k++) {
        lsg_hit_small const & h = out.small[k];
        size_t const N = h.x >> logB;
        if (N >= nreg || idx[N] < 0) continue;
        if (!at_survivor(regions[idx[N]], h.x & mask)) continue;
        bucket_update_t<1, primehint_t> u;
        u.p = h.p;
        u.x = h.x & mask;
        regions[idx[N]].primes[h.side].push_back(u);
    }
    std::lock_guard<std::mutex> const dummy(stats_lock);
    stats_wall += wct_seconds() - t0;
    stats_calls++;
    stats_survivors += out.nsurv;
    /* the GPU drops the rejected survivors (out.nrejected); pad is set
     * on the ones it sends only if that is turned off */
    nrejected += out.nrejected;
    stats_cof_rejected += nrejected;

    return true;
}

bool las_sieve_gpu_run(nfs_work & ws, qlattice_basis const & Q)
{
    if (!check_mode() || !applicable(ws, Q)) return false;
    lsg_ctx * ctx = context();
    if (!ctx) return false;
    gpu_inputs in;
    if (!build_inputs(ws, Q, in)) return false;
    int const nsides = 2;

    auto C = std::make_shared<check_data>();
    size_t const area = (size_t) ws.J << ws.conf.logI;
    uint8_t * S[2];
    for (int side = 0; side < 2; side++) {
        C->S[side].resize(area);
        S[side] = C->S[side].data();
    }
    C->logI = ws.conf.logI;
    C->small[0] = in.small[0];
    C->small[1] = in.small[1];
    double const t0 = wct_seconds();
    lsg_sieve(ctx, &in.qi, in.sd, nsides, S, in.bound, nullptr, nullptr, 0, nullptr);
    {
        std::lock_guard<std::mutex> const dummy(stats_lock);
        stats_wall += wct_seconds() - t0;
        stats_calls++;
    }
    std::lock_guard<std::mutex> const dummy(check_lock);
    checks[&ws] = C;
    return true;
}

void las_sieve_gpu_check_region(nfs_work const & ws, int side, unsigned int N,
        unsigned char const * S)
{
    if (!check_mode() || !S) return;
    std::shared_ptr<check_data> C;
    {
        std::lock_guard<std::mutex> const dummy(check_lock);
        auto it = checks.find(&ws);
        if (it == checks.end()) return;
        C = it->second;
    }
    int const logI = C->logI;
    int const logB = LOG_BUCKET_REGION;
    if (logI > logB) return;    /* regions are whole rows here */
    size_t const base = (size_t) N << logB;
    if (base + ((size_t) 1 << logB) > C->S[side].size()) return;
    int const I = 1 << logI;
    unsigned long cmp = 0, bad = 0;
    for (size_t x = 0; x < ((size_t) 1 << logB); x++) {
        size_t const xg = base + x;
        long const i = (long) (xg & (I - 1)) - I / 2;
        unsigned long const j = xg >> logI;
        bool coprime;
        if (j == 0) coprime = (i == 1);
        else {
            unsigned long a = i < 0 ? -i : i, b = j;
            while (b) { unsigned long t = a % b; a = b; b = t; }
            coprime = a == 1;
        }
        if (!coprime) continue;
        cmp++;
        if (S[x] != C->S[side][xg]) {
            if (n_mismatch + bad < 10) {
                fprintf(stderr, "# GPU sieve mismatch: side %d (i,j)=(%ld,%lu)"
                        " CPU %u GPU %u\n", side, i, j, S[x], C->S[side][xg]);
                if (n_mismatch + bad < 2 && getenv("CADO_LAS_GPU_SIEVE_DEBUG")) {
                    unsigned sum = 0;
                    for (auto const & e : C->small[side]) {
                        unsigned long const ir = (unsigned long) (i + I / 2);
                        bool hit;
                        if (e.kind == 0) hit = (ir % e.q) == ((unsigned long) e.r * j + I / 2) % e.q;
                        else hit = (j % e.g == 0) && (ir % e.q) == ((unsigned long) e.r * (j / e.g) + I / 2) % e.q;
                        if (hit) {
                            sum += e.logp;
                            fprintf(stderr, "#   small hit: kind %d q=%u r=%u g=%u logp=%u\n",
                                    e.kind, e.q, e.r, e.g, e.logp);
                        }
                    }
                    fprintf(stderr, "#   small total %u\n", sum);
                }
            }
            bad++;
        }
    }
    n_compared += cmp;
    n_mismatch += bad;
    n_regions++;
}

void las_sieve_gpu_report()
{
    if (real_mode() && stats_calls) {
        verbose_fmt_print(0, 1, "# GPU siever: {} special-q, {:.2f} s"
                " (GPU and copies), {} survivors, {} rejected by the GPU"
                " cofactor filter\n", stats_calls, stats_wall,
                stats_survivors, stats_cof_rejected);
        return;
    }
    if (!check_mode()) return;
    verbose_fmt_print(0, 1, "# GPU sieve check: {} side-regions, {} coprime"
            " positions compared, {} mismatches; GPU {} calls, {:.2f} s\n",
            n_regions.load(), n_compared.load(), n_mismatch.load(),
            stats_calls, stats_wall);
}

#else

bool las_sieve_gpu_enabled(nfs_work const &, qlattice_basis const &) { return false; }
bool las_sieve_gpu_sieve(nfs_work &, qlattice_basis const &,
        std::vector<gpu_sieve_region> &, nfs_work_cofac const *) { return false; }
bool las_sieve_gpu_run(nfs_work &, qlattice_basis const &) { return false; }
void las_sieve_gpu_check_region(nfs_work const &, int, unsigned int,
        unsigned char const *) {}
void las_sieve_gpu_report() {}

#endif

/* vim: set sw=4: */
