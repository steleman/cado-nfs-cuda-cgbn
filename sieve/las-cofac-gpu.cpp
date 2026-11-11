#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ecm/facul-gpu.hpp"
#include "las-auxiliary-data.hpp"
#include "las-cofac-gpu.hpp"
#include "las-cofac-standalone.hpp"
#include "las-detached-cofac.hpp"
#include "las-threads-work-data.hpp"
#include "threadpool.hpp"
#include "timing.h"
#include "verbose.hpp"

struct las_cofac_gpu::impl {
    facul_gpu * g = nullptr;

    struct entry {
        thread_pool * pool;
        std::shared_ptr<nfs_work_cofac> wc;
        std::shared_ptr<nfs_aux> aux;
        std::shared_ptr<cofac_standalone> cur;
    };

    std::mutex m;
    std::condition_variable cv;         /* the batch thread waits */
    std::condition_variable done_cv;    /* flush() waits */
    std::vector<entry> pending;
    double oldest = 0;
    std::map<thread_pool *, size_t> outstanding;
    int flushing = 0;
    size_t batch = 65536;
    double max_wait = 1.0;
    std::thread th;

    /* statistics */
    unsigned long ncand = 0, ncpu = 0, nbatches = 0;
    double busy = 0;

    void loop();
    void process(std::vector<entry> & b);
};

las_cofac_gpu * las_cofac_gpu::get()
{
    static std::once_flag once;
    static las_cofac_gpu * instance = nullptr;
    std::call_once(once, []() {
        facul_gpu * g = facul_gpu::get();
        if (!g) return;
        auto * l = new las_cofac_gpu;
        l->d->g = g;
        if (const char * e = getenv("CADO_LAS_GPU_BATCH"))
            l->d->batch = std::max(1, atoi(e));
        if (const char * e = getenv("CADO_LAS_GPU_WAIT"))
            l->d->max_wait = atof(e);
        l->d->th = std::thread([l]() { l->d->loop(); });
        l->d->th.detach();
        verbose_fmt_print(0, 1, "# GPU cofactorization on device {},"
                " batches of {}\n", g->device(), l->d->batch);
        instance = l;
    });
    return instance;
}

las_cofac_gpu::las_cofac_gpu() : d(new impl) {}
las_cofac_gpu::~las_cofac_gpu() = default;

void las_cofac_gpu::submit(thread_pool & pool,
        std::shared_ptr<nfs_work_cofac> const & wc,
        std::shared_ptr<nfs_aux> const & aux,
        cofac_standalone && cur)
{
    std::lock_guard<std::mutex> const lk(d->m);
    if (d->pending.empty()) d->oldest = wct_seconds();
    d->pending.push_back({ &pool, wc, aux,
            std::make_shared<cofac_standalone>(std::move(cur)) });
    d->outstanding[&pool]++;
    if (d->pending.size() >= d->batch) d->cv.notify_one();
}

void las_cofac_gpu::flush(thread_pool & pool)
{
    std::unique_lock<std::mutex> lk(d->m);
    d->flushing++;
    d->cv.notify_one();
    d->done_cv.wait(lk, [&]() { return d->outstanding[&pool] == 0; });
    d->flushing--;
}

void las_cofac_gpu::impl::loop()
{
    std::unique_lock<std::mutex> lk(m);
    for (;;) {
        cv.wait_for(lk, std::chrono::milliseconds(20), [&]() {
                return pending.size() >= batch || (flushing && !pending.empty());
        });
        if (pending.empty()) continue;
        if (pending.size() < batch && !flushing && wct_seconds() - oldest < max_wait)
            continue;
        std::vector<entry> b;
        b.swap(pending);
        lk.unlock();
        double const t0 = wct_seconds();
        process(b);
        double const t1 = wct_seconds();
        lk.lock();
        busy += t1 - t0;
        nbatches++;
        for (auto const & e : b) outstanding[e.pool]--;
        done_cv.notify_all();
    }
}

void las_cofac_gpu::impl::process(std::vector<entry> & b)
{
    /* one strategy book per call of factor_all */
    std::map<facul_strategies const *, std::vector<size_t>> groups;
    for (size_t i = 0; i < b.size(); i++)
        groups[b[i].wc->strategies].push_back(i);

    std::vector<facul_gpu_outcome> out;
    std::vector<std::vector<cxx_mpz> const *> ptrs;
    unsigned long cpu = 0;
    for (auto const & gr : groups) {
        ptrs.clear();
        for (size_t i : gr.second) ptrs.push_back(&b[i].cur->norm);
        if (gr.first) {
            g->factor_all(ptrs, *gr.first, out);
        } else {
            out.assign(ptrs.size(), facul_gpu_outcome {});
            for (auto & o : out) o.cpu = true;
        }
        for (size_t k = 0; k < gr.second.size(); k++) {
            entry & e = b[gr.second[k]];
            cpu += out[k].cpu;
            e.pool->add_task(thread_pool::QUEUE_ECM,
                    [wc = e.wc, aux = e.aux, cur = e.cur,
                     o = std::make_shared<facul_gpu_outcome>(std::move(out[k]))]
                    (worker_thread * w) {
                        detached_cofac(w, *wc, *aux, std::move(*cur), o.get());
                    });
        }
    }
    std::lock_guard<std::mutex> const lk(m);
    ncand += b.size();
    ncpu += cpu;
}

void las_cofac_gpu::report()
{
    las_cofac_gpu * l = get();
    if (!l) return;
    double ks;
    unsigned long nj;
    l->d->g->stats(ks, nj);
    std::lock_guard<std::mutex> const lk(l->d->m);
    verbose_fmt_print(0, 1, "# GPU cofactorization: {} candidates ({} on the"
            " CPU), {} batches, {} GPU jobs, kernels {:.2f} s, batch thread"
            " busy {:.2f} s\n", l->d->ncand, l->d->ncpu, l->d->nbatches, nj,
            ks, l->d->busy);
}

/* vim: set sw=4: */
