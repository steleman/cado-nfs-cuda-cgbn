#include "cado.h" // IWYU pragma: keep

/* Top-level bucket fill on the GPU (las-fill-gpu-kernels.cu).
 *
 * fill_in_buckets_toplevel_gpu() replaces the per-slice tasks of
 * fill_in_buckets_one_side when the top level is 1 and there are no
 * sublattices: the GPU computes the updates of all the side's top-level
 * slices, then one task per slice copies its updates into a bucket array
 * (as the CPU task of that slice would have written them) and handles
 * the lattices that are left to the host: the projective-like and
 * vertical ones (with the code of fill_in_buckets_toplevel_impl) and the
 * powers of 2.
 *
 * Off with CADO_LAS_GPU_FILL=0. (There is no check mode of its own:
 * it was checked by comparing las's relations with the CPU fill's.)
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <gmp.h>

#include "bucket.hpp"
#include "bucket-push-update.hpp"       // IWYU pragma: keep
#include "fb.hpp"
#include "las-auxiliary-data.hpp"
#include "las-config.hpp"
#include "las-fill-gpu.hpp"
#include "las-plattice.hpp"
#include "las-qlattice.hpp"
#include "las-threads-work-data.hpp"
#include "las-where-am-i-proxy.hpp"
#include "las-where-am-i.hpp"
#include "macros.h"
#include "tdict.hpp"
#include "threadpool.hpp"
#include "timing.h"
#include "verbose.hpp"

#if defined(HAVE_CUDA) && !defined(SIQS_SIEVE) && !defined(SUPPORT_LARGE_Q) && !defined(TRACE_K)
#define LAS_FILL_GPU 1
#include "las-fill-gpu-kernels.h"

namespace {

/* the entries the host keeps: powers of 2 */
struct host_entry {
    fb_entry_general const * e;
    slice_offset_t hint;
};

struct side_table {
    lfg_table * t = nullptr;
    std::vector<slice_index_t> slice_index;     /* local -> global */
    std::vector<std::vector<host_entry>> host;  /* per local slice */
    ~side_table() { lfg_table_free(t); }
};

std::mutex tables_lock;
std::map<std::pair<void const *, int>, std::shared_ptr<side_table>> tables;

bool enabled()
{
    static int const e = []() {
        const char * s = getenv("CADO_LAS_GPU_FILL");
        return s ? atoi(s) : 1;
    }();
    return e != 0;
}

/* one context per subjob thread and side */
lfg_ctx * context(int side)
{
    static thread_local lfg_ctx * ctx[2] = { nullptr, nullptr };
    static thread_local bool tried[2] = { false, false };
    if (!tried[side]) {
        tried[side] = true;
        ctx[side] = lfg_ctx_create();
    }
    return ctx[side];
}

std::mutex stats_lock;
double stats_wall = 0;
unsigned long stats_calls = 0;
size_t stats_updates = 0;

std::shared_ptr<side_table> get_table(nfs_work const & ws, int side)
{
    nfs_work::side_data const & wss(ws.sides[side]);
    std::lock_guard<std::mutex> const dummy(tables_lock);
    auto key = std::make_pair((void const *) wss.fbs, side);
    auto it = tables.find(key);
    if (it != tables.end()) return it->second;

    auto T = std::make_shared<side_table>();
    std::vector<lfg_lat> lats;
    fb_factorbase::slicing::part const & P = wss.fbs->get_part(1);
    uint32_t ls = 0;
    bool ok = true;
    P.foreach_slice([&](auto const & s) {
        using E = std::decay_t<decltype(s)>::entry_t;
        T->slice_index.push_back(s.get_index());
        T->host.emplace_back();
        slice_offset_t i_entry = 0;
        for (auto const & e : s) {
            if constexpr (E::is_general_type) {
                if (!(e.q & 1)) {
                    T->host.back().push_back({ &e, i_entry });
                    i_entry++;
                    continue;
                }
#ifndef BUCKET_SIEVE_POWERS
                if (e.k > 1) { i_entry++; continue; }
#endif
            }
            for (int i = 0; i < e.get_nr_roots(); i++) {
                lfg_lat L {};
                L.q = e.get_q();
                L.p = e.p;
                L.r = e.get_r(i);
                L.hint = i_entry;
                L.slice = ls;
                L.proj = e.get_proj(i);
                lats.push_back(L);
            }
            i_entry++;
        }
        ls++;
        if (ls > 65535) ok = false;
    });
    if (!ok) return nullptr;
    T->t = lfg_table_create(lats.data(), lats.size(), ls);
    verbose_fmt_print(0, 2, "# GPU fill: side {}, {} top-level slices,"
            " {} lattices\n", side, ls, lats.size());
    tables[key] = T;
    return T;
}

/* The lattices of fill_in_buckets_toplevel_impl that the GPU leaves to
 * the host: same code. */
template <hint_type HINT>
void host_lattice(bucket_array_t<1, HINT> & BA, nfs_work const & ws,
        plattice_info const & pli, slice_offset_t hint,
        slice_index_t slice_index, where_am_I & w)
{
    int const logI = ws.conf.logI;
    plattice_enumerator::fence const F(logI, ws.J);
    int const logB = LOG_BUCKET_REGIONS[1];
    uint32_t const maskB1I = (UINT32_C(1) << std::min(logB, logI)) - 1;
    typename bucket_array_t<1, HINT>::update_t::br_index_t const bmask =
        (1UL << logB) - 1;

    plattice_enumerator ple(pli, hint, logI);
    ple.next(F);
    if (pli.is_discarded()) return;
    typename bucket_array_t<1, HINT>::update_t u(0, 0, hint, slice_index);
    if (UNLIKELY(ple.is_projective_like(logI))) {
        while (!ple.done(F)) {
            u.set_x(ple.get_x() & bmask);
            int const N = ple.get_x() >> logB;
            int const n = ple.advance_to_end_of_row_or_smallest_region(maskB1I);
            BA.push_row_update(slice_index, ple.get_inc_step(), N, n, u, w);
            ple.next(F);
        }
    } else if (UNLIKELY(pli.is_vertical_line(logI))) {
        if (!ple.done(F)) {
            u.set_x(ple.get_x() & bmask);
            BA.push_update(ple.get_x() >> logB, u, w);
            ple.finish();
        }
    } else {
        while (!ple.done(F)) {
            if (LIKELY(ple.probably_coprime(F))) {
                u.set_x(ple.get_x() & bmask);
                BA.push_update(ple.get_x() >> logB, u, w);
            }
            ple.next(F);
        }
    }
}

/* the in-memory layout of an update, checked once */
template <hint_type HINT> bool layout_ok()
{
    using U = typename bucket_array_t<1, HINT>::update_t;
    U u(0x1234, 0, 0x5678, 0);
    if constexpr (std::is_same_v<HINT, shorthint_t>) {
        uint32_t v;
        static_assert(sizeof(U) == 4);
        memcpy(&v, &u, 4);
        return v == (0x5678u | (0x1234u << 16));
    } else {
        uint16_t v;
        static_assert(sizeof(U) == 2);
        memcpy(&v, &u, 2);
        return v == 0x1234u;
    }
}

template <hint_type HINT>
bool fill_side(nfs_work & ws, nfs_aux & aux, qlattice_basis const & Q,
        thread_pool & pool, int side, where_am_I & w)
{
    static bool const layout = layout_ok<HINT>();
    if (!layout) return false;
    nfs_work::side_data & wss(ws.sides[side]);
    int const logB = LOG_BUCKET_REGIONS[1];
    if (logB > 16 || ws.nb_buckets[1] > 8192 || ws.conf.logI > 16)
        return false;
    lfg_ctx * ctx = context(side);
    if (!ctx) return false;
    auto T = get_table(ws, side);
    if (!T) return false;

    lfg_qinfo qi {};
    qi.a0 = Q.a0; qi.b0 = Q.b0; qi.a1 = Q.a1; qi.b1 = Q.b1;
    if (Q.doing.is_prime()) {
        if (mpz_fits_ulong_p(Q.doing.p))
            qi.qfac[qi.nqfac++] = mpz_get_ui(Q.doing.p);
    } else {
        for (auto f : Q.doing.prime_factors) {
            if (qi.nqfac == 8) return false;
            qi.qfac[qi.nqfac++] = f;
        }
    }
    qi.logI = ws.conf.logI;
    qi.J = ws.J;
    qi.logB = logB;
    qi.nbuckets = ws.nb_buckets[1];
    qi.hint_bytes = std::is_same_v<HINT, shorthint_t> ? 2 : 0;

    const uint32_t * counts;
    const void * updates;
    size_t nupd, nspec;
    const lfg_special * spec;
    double const t0 = wct_seconds();
    lfg_fill(ctx, T->t, &qi, &counts, &updates, &nupd, &spec, &nspec);
    {
        std::lock_guard<std::mutex> const dummy(stats_lock);
        stats_wall += wct_seconds() - t0;
        stats_calls++;
        stats_updates += nupd;
    }

    using update_t = typename bucket_array_t<1, HINT>::update_t;
    uint32_t const nb = ws.nb_buckets[1];
    size_t const ns = T->slice_index.size();

    /* where each slice's updates start, and its special lattices */
    auto starts = std::make_shared<std::vector<size_t>>(ns + 1, 0);
    for (size_t s = 0; s < ns; s++) {
        size_t sum = 0;
        for (uint32_t b = 0; b < nb; b++) sum += counts[s * nb + b];
        (*starts)[s + 1] = (*starts)[s] + sum;
    }
    auto specials = std::make_shared<std::vector<std::vector<lfg_special>>>(ns);
    for (size_t k = 0; k < nspec; k++)
        (*specials)[spec[k].slice].push_back(spec[k]);

    for (size_t s = 0; s < ns; s++) {
        pool.add_task(thread_pool::QUEUE_GENERIC, 0.0,
            [&ws, &aux, &Q, side, s, T, starts, specials, counts, updates, nb]
            (worker_thread * worker) {
                int const id = worker->rank();
                timetree_t & timer(aux.get_timer(worker));
                ENTER_THREAD_TIMER(timer);
                MARK_TIMER_FOR_SIDE(timer, side);
#ifndef DISABLE_TIMINGS
                timetree_t::accounting_child const local_timer_sentry(timer,
                        tdict_slot_for_fibt);
#endif
                where_am_I & w(aux.th[id].w);
                nfs_work::side_data & wss(ws.sides[side]);
                auto acquired = wss.reserve_BA<1, HINT>();
                bucket_array_t<1, HINT> & BA = acquired.access();
                slice_index_t const si = T->slice_index[s];
                BA.add_slice_index(si);
                BA.push_block((update_t const *) updates + (*starts)[s],
                        counts + s * nb);
                int const logI = ws.conf.logI;
                for (auto const & sp : (*specials)[s]) {
                    plattice_info const pli(sp.q, sp.r, sp.proj, logI);
                    host_lattice(BA, ws, pli, sp.hint, si, w);
                }
                /* powers of 2, as fill_in_buckets_toplevel_impl */
                for (auto const & he : T->host[s]) {
                    fb_entry_general const & e = *he.e;
                    if (!Q.is_coprime_to(e.p)) continue;
                    fb_entry_general transformed;
                    e.transform_roots(transformed, Q);
                    for (unsigned char i = 0; i != transformed.nr_roots; i++) {
                        plattice_info const pli(transformed.get_q(),
                                transformed.get_r(i), transformed.get_proj(i),
                                logI);
                        host_lattice(BA, ws, pli, he.hint, si, w);
                    }
                }
            });
    }
    (void) wss;
    (void) w;
    return true;
}

} /* namespace */

bool fill_in_buckets_toplevel_gpu(nfs_work & ws, nfs_aux & aux,
        qlattice_basis const & Q, thread_pool & pool, int side,
        where_am_I & w)
{
    if (!enabled()) return false;
    if (ws.toplevel != 1 || Q.sublat.m > 1 || side > 1) return false;
    if (ws.sides[side].fbs->get_toplevel() < 1) return false;
    if (ws.conf.needs_resieving())
        return fill_side<shorthint_t>(ws, aux, Q, pool, side, w);
    else
        return fill_side<emptyhint_t>(ws, aux, Q, pool, side, w);
}

void las_fill_gpu_report()
{
    std::lock_guard<std::mutex> const dummy(stats_lock);
    if (!stats_calls) return;
    verbose_fmt_print(0, 1, "# GPU bucket fill: {} side fills, {:.2f} s"
            " (GPU and copies), {} updates\n", stats_calls, stats_wall,
            stats_updates);
}

#else

bool fill_in_buckets_toplevel_gpu(nfs_work &, nfs_aux &,
        qlattice_basis const &, thread_pool &, int, where_am_I &)
{
    return false;
}
void las_fill_gpu_report() {}

#endif

/* vim: set sw=4: */
