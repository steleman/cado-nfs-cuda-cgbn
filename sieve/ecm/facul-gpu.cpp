#include "cado.h" // IWYU pragma: keep

/* Host side of the GPU cofactorization (see facul-gpu.hpp): turns the
 * plans of facul's methods into the blob of facul-gpu-data.h.
 *
 * GMP is used here only to build that blob, once per method: the P-1
 * exponent, and the ECM curves as rationals (b = (A+2)/4 and the
 * starting point, computed over Q from the parameterization; the GPU
 * reduces them modulo each cofactor). Each curve is checked against
 * CADO's own parameterization code modulo 2^61-1.
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

#include "cxx_mpz.hpp"
#include "facul-gpu.hpp"
#include "gmp_aux.h"
#include "macros.h"

#ifdef HAVE_CUDA

#include "arithxx/modredc64.hpp"
#include "bytecode.h"
#include "ec_arith_common.hpp"
#include "ec_arith_cost.h"
#include "ec_parameterization.hpp"
#include "facul-gpu-kernels.h"
#include "facul_ecm.h"
#include "facul.hpp"
#include "facul_method.hpp"
#include "facul_strategies.hpp"
#include "pm1.hpp"
#include "pp1.hpp"
#include "stage2.h"

namespace {

/* {{{ rationals and Weierstrass curves over Q */
struct Q {
    mpq_t q;
    Q() { mpq_init(q); }
    explicit Q(long n, unsigned long d = 1) { mpq_init(q); mpq_set_si(q, n, d); mpq_canonicalize(q); }
    Q(Q const & o) { mpq_init(q); mpq_set(q, o.q); }
    Q & operator=(Q const & o) { mpq_set(q, o.q); return *this; }
    ~Q() { mpq_clear(q); }
};
Q operator+(Q const & a, Q const & b) { Q r; mpq_add(r.q, a.q, b.q); return r; }
Q operator-(Q const & a, Q const & b) { Q r; mpq_sub(r.q, a.q, b.q); return r; }
Q operator*(Q const & a, Q const & b) { Q r; mpq_mul(r.q, a.q, b.q); return r; }
Q operator/(Q const & a, Q const & b) { Q r; mpq_div(r.q, a.q, b.q); return r; }
Q operator*(long k, Q const & a) { return Q(k) * a; }
Q sq(Q const & a) { return a * a; }
Q cube(Q const & a) { return a * a * a; }

/* k*P on y^2 = x^3 + a*x + b, affine, k >= 1 */
void weierstrass_smul(Q & x, Q & y, Q const & a, Q const & x0, Q const & y0,
        unsigned long k)
{
    Q X = x0, Y = y0;
    int top = 63;
    while (!((k >> top) & 1)) top--;
    for (int i = top - 1; i >= 0; i--) {
        /* double */
        Q const l = (3 * sq(X) + a) / (2 * Y);
        Q const X2 = sq(l) - 2 * X;
        Y = l * (X - X2) - Y;
        X = X2;
        if ((k >> i) & 1) {
            Q const l2 = (Y - y0) / (X - x0);
            Q const X3 = sq(l2) - X - x0;
            Y = l2 * (x0 - X3) - y0;
            X = X3;
        }
    }
    x = X;
    y = Y;
}

/* b = (A+2)/4 and the starting point (x0::z0) of the curve, over Q, as
 * ec_parameterization.hpp computes them modulo n. */
bool curve_over_Q(ec_parameterization_t param, unsigned long k, Q & b,
        Q & x0, Q & z0)
{
    switch (param) {
        case BRENT12: {
            Q const s(k);
            Q const u = sq(s) - Q(5), v = 4 * s;
            x0 = cube(u);
            z0 = cube(v);
            b = cube(v - u) * (3 * u + v) / (16 * cube(u) * v);
            return true;
        }
        case MONTY12: {
            Q u, v;
            weierstrass_smul(u, v, Q(-12), Q(-2), Q(4), k);
            Q const uu = sq(u), vv = sq(v);
            Q const d2 = vv - 4 * uu, d1 = vv + 12 * uu;
            x0 = 2 * uu * sq(uu + Q(12));
            z0 = 2 * d1 * d2;
            b = 1024 * vv * cube(uu) / (d1 * cube(d2));
            return true;
        }
        case MONTY16:
            x0 = Q(8);
            z0 = Q(15);
            b = Q(83521, 57600);
            return true;
        case MONTYTWED12: {
            Q Tx, Ty;
            weierstrass_smul(Tx, Ty, Q(-9747), Q(15), Q(378), k);
            Q const U = 144 * (Tx + Q(3)), W(2985984);
            Q const u0 = 96 * U, u1 = W - u0, u2 = sq(u1), u3 = sq(u0);
            Q const u4 = u2 - 5 * u3, u5 = cube(u4), u7 = 4 * u1;
            x0 = u5 * u3 * u0;
            z0 = cube(u7 * u3);
            Q const u2b = u3 * u0;
            Q const Tt = Q(1) / (16 * u3 * u5 * u7 * u2b);
            Q const Tz = u4 * u0;
            Q const Ty1 = u7 * u3;
            Q const Tx1 = 3 * Tz + Ty1;
            Q const Ty2 = Ty1 - Tz;
            b = cube(Ty2) * Tx1 * Tt;
            return true;
        }
        default:
            return false;
    }
}

/* x mod p for a rational x (p prime, den coprime to p) */
uint64_t q_mod(Q const & x, uint64_t p)
{
    cxx_mpz n, d, P;
    mpz_set_ui(P, p);
    mpz_mod(n, mpq_numref(x.q), P);
    mpz_mod(d, mpq_denref(x.q), P);
    mpz_invert(d, d, P);
    mpz_mul(n, n, d);
    mpz_mod(n, n, P);
    return mpz_get_ui(n);
}

/* The curve over Q against CADO's parameterization modulo 2^61-1 */
bool curve_check(ec_parameterization_t param, unsigned long k, Q const & b,
        Q const & x0, Q const & z0)
{
    using layer = arithxx_modredc64;
    uint64_t const p = (UINT64_C(1) << 61) - 1;
    layer::Modulus const m { Integer64(p) };
    layer::Residue bb(m);
    ec_point<layer> P(m);
    int r;
    switch (param) {
        case BRENT12: r = ec_parameterization_Brent_Suyama<layer>(bb, P, k, m); break;
        case MONTY12: r = ec_parameterization_Montgomery12<layer>(bb, P, k, m); break;
        case MONTY16: r = ec_parameterization_Montgomery16<layer>(bb, P, k, m); break;
        case MONTYTWED12: r = ec_parameterization_Z6<layer>(&bb, P, k, MONTGOMERY_xz, m); break;
        default: return false;
    }
    if (!r) return false;
    cxx_mpz const P_(p);
    uint64_t const cb = uint64_t(m.get(bb));
    uint64_t const cx = uint64_t(m.get(P.x));
    uint64_t const cz = uint64_t(m.get(P.z));
    uint64_t const gb = q_mod(b, p), gx = q_mod(x0, p), gz = q_mod(z0, p);
    /* x0/z0 == cx/cz  <=>  x0*cz == cx*z0 */
    cxx_mpz l, rr;
    mpz_set_ui(l, gx); mpz_mul_ui(l, l, cz); mpz_mod(l, l, P_);
    mpz_set_ui(rr, cx); mpz_mul_ui(rr, rr, gz); mpz_mod(rr, rr, P_);
    return cb == gb && mpz_cmp(l, rr) == 0;
}
/* }}} */

size_t prac_len(bytecode_const bc)
{
    size_t i = 0;
    while (bc[i] != PRAC_BLOCK_FINAL) i++;
    return i + 1;
}

} /* namespace */

facul_gpu * facul_gpu::get()
{
    static std::once_flag once;
    static facul_gpu * instance = nullptr;
    std::call_once(once, []() {
        const char * e = getenv("CADO_LAS_GPU_COFAC");
        if (e && atoi(e) == 0) return;
        fgpu_ctx * c = fgpu_create(-1);
        if (c) instance = new facul_gpu(c);
    });
    return instance;
}

facul_gpu::facul_gpu(fgpu_ctx * ctx) : ctx(ctx) {}

facul_gpu::~facul_gpu()
{
    fgpu_free(ctx);
}

int facul_gpu::device() const { return fgpu_device(ctx); }

void facul_gpu::stats(double & kernel_seconds, unsigned long & njobs) const
{
    fgpu_stats(ctx, &kernel_seconds, &njobs);
}

fgpu_bigint facul_gpu::bigint(cxx_mpz const & z)
{
    fgpu_bigint r {};
    r.sign = mpz_sgn(z);
    r.off = p64.size();
    size_t const n = mpz_size(z);
    for (size_t i = 0; i < n; i++)
        p64.push_back(mpz_getlimbn_uint64(z, i));
    r.len = n;
    return r;
}

facul_gpu::method_entry const & facul_gpu::method(facul_method const * fm)
{
    auto it = method_cache.find(fm);
    if (it != method_cache.end()) return it->second;

    method_entry e {};
    e.ok = true;
    fgpu_method & g = e.m;
    stage2_plan_t const * s2 = nullptr;

    switch (fm->method) {
        case PM1_METHOD: {
            auto const & p = *(pm1_plan_t const *) fm->plan;
            g.type = FGPU_PM1;
            g.B1 = p.B1;
            g.exp2 = p.exp2;
            fgpu_bigint const E = bigint(p.E);
            g.E_off = E.off;
            g.E_len = E.len;
            s2 = &p.stage2;
            break;
        }
        case PP1_27_METHOD:
        case PP1_65_METHOD: {
            auto const & p = *(pp1_plan_t const *) fm->plan;
            g.type = FGPU_PP1;
            g.B1 = p.B1;
            g.exp2 = p.exp2;
            size_t const len = p.bc ? prac_len(p.bc) : 0;
            g.bc_off = p8.size();
            g.bc_len = len;
            if (len) p8.insert(p8.end(), p.bc, p.bc + len);
            g.c[0] = bigint(cxx_mpz(fm->method == PP1_27_METHOD ? 2 : 6));
            g.c[1] = bigint(cxx_mpz(fm->method == PP1_27_METHOD ? 7 : 5));
            s2 = &p.stage2;
            break;
        }
        case EC_METHOD: {
            auto const & p = *(ecm_plan_t const *) fm->plan;
            g.type = FGPU_ECM;
            g.B1 = p.B1;
            g.exp2 = p.exp2;
            Q b, x0, z0;
            if (!curve_over_Q(p.parameterization, p.parameter, b, x0, z0)
                    || !curve_check(p.parameterization, p.parameter, b, x0, z0)) {
                fprintf(stderr, "facul-gpu: no GPU curve for parameterization"
                        " %d, parameter %lu\n", int(p.parameterization), p.parameter);
                e.ok = false;
                break;
            }
            g.c[0] = bigint(cxx_mpz(mpq_numref(b.q)));
            g.c[1] = bigint(cxx_mpz(mpq_denref(b.q)));
            /* (x0 :: z0) as integers */
            cxx_mpz X, Z;
            mpz_mul(X, mpq_numref(x0.q), mpq_denref(z0.q));
            mpz_mul(Z, mpq_numref(z0.q), mpq_denref(x0.q));
            g.c[2] = bigint(X);
            g.c[3] = bigint(Z);
            /* stage 1: the plan's PRAC code, or for the twisted Edwards
             * curves a PRAC code for the same multiplier */
            bytecode bc = p.bc;
            bytecode own = nullptr;
            if (p.parameterization & FULLMONTYTWED) {
                cxx_mpz k_mm, k_prac;
                prac_cost_t const cost = { MONTGOMERY_DBL, MONTGOMERY_dADD };
                bool same = !bytecode_mishmash_scalar(p.bc, k_mm);
                /* the extra 3 of the torsion 12 curves, if the plan has it */
                for (unsigned int pow3_extra = 1; same; pow3_extra--) {
                    bytecode_prac_encode(&own, p.B1, p.exp2, pow3_extra, &cost, 1, 0);
                    if (!bytecode_prac_scalar(own, k_prac) && mpz_cmp(k_mm, k_prac) == 0)
                        break;
                    free(own);
                    own = nullptr;
                    if (pow3_extra == 0) same = false;
                }
                if (!same) {
                    fprintf(stderr, "facul-gpu: no PRAC chain for the"
                            " stage 1 multiplier of B1=%u\n", p.B1);
                    free(own);
                    e.ok = false;
                    break;
                }
                bc = own;
            }
            size_t const len = bc ? prac_len(bc) : 0;
            g.bc_off = p8.size();
            g.bc_len = len;
            if (len) p8.insert(p8.end(), bc, bc + len);
            free(own);
            s2 = &p.stage2;
            break;
        }
        default:
            e.ok = false;
    }

    if (e.ok && s2 && s2->B2 > g.B1 && s2->w) {
        g.B2 = s2->B2;
        g.w = s2->w;
        g.U_len = s2->U_len;
        g.vmin = s2->vmin;
        g.vmax = s2->vmax;
        ASSERT_ALWAYS(g.U_len <= FGPU_UMAX);
        g.U_off = p32.size();
        p32.insert(p32.end(), s2->U, s2->U + s2->U_len);
        g.pairs_off = p16.size();
        for (unsigned int const * q = s2->pairs;; q++) {
            if (*q == PAIR_END) { p16.push_back(FGPU_PAIR_END); break; }
            if (*q == PAIR_INCR_V) { p16.push_back(FGPU_PAIR_INCR_V); continue; }
            ASSERT_ALWAYS(*q < s2->U_len);
            p16.push_back((uint16_t) *q);
        }
    } else if (e.ok && s2) {
        g.B2 = s2->B2;
        g.w = 0;
    }

    return method_cache.emplace(fm, e).first->second;
}

int facul_gpu::chain(std::vector<facul_method_side> const & meths, int side,
        unsigned long lpb, double BB, double BBB)
{
    std::lock_guard<std::mutex> const dummy(lock);
    auto key = std::make_pair((void const *) &meths, side);
    auto it = chain_cache.find(key);
    if (it != chain_cache.end()) return it->second;

    std::vector<fgpu_method> ms;
    bool ok = true;
    for (auto const & m : meths) {
        if (m.side != side) continue;
        auto const & e = method(m.method);
        if (!e.ok) { ok = false; break; }
        fgpu_method g = e.m;
        g.flags = (m.stop_unsplit ? FGPU_FLAG_STOP_UNSPLIT : 0)
                | (m.is_last ? FGPU_FLAG_LAST : 0);
        ms.push_back(g);
    }
    int id = -1;
    if (ok && !ms.empty()) {
        fgpu_chain c {};
        c.m0 = methods.size();
        c.nm = ms.size();
        c.lpb = lpb;
        c.BB = BB;
        c.BBB = BBB;
        methods.insert(methods.end(), ms.begin(), ms.end());
        id = chains.size();
        chains.push_back(c);
        dirty = true;
    }
    chain_cache[key] = id;
    return id;
}

void facul_gpu::upload()
{
    auto align = [](size_t x) { return (x + 15) & ~size_t(15); };
    fgpu_blob_header h {};
    h.nchains = chains.size();
    h.nmethods = methods.size();
    size_t off = align(sizeof(h));
    h.off_chains = off; off = align(off + chains.size() * sizeof(fgpu_chain));
    h.off_methods = off; off = align(off + methods.size() * sizeof(fgpu_method));
    h.off64 = off; off = align(off + p64.size() * 8);
    h.off32 = off; off = align(off + p32.size() * 4);
    h.off16 = off; off = align(off + p16.size() * 2);
    h.off8 = off; off = align(off + p8.size());
    h.n64 = p64.size(); h.n32 = p32.size(); h.n16 = p16.size(); h.n8 = p8.size();
    std::vector<uint8_t> blob(off, 0);
    memcpy(blob.data(), &h, sizeof(h));
    if (!chains.empty()) memcpy(blob.data() + h.off_chains, chains.data(), chains.size() * sizeof(fgpu_chain));
    if (!methods.empty()) memcpy(blob.data() + h.off_methods, methods.data(), methods.size() * sizeof(fgpu_method));
    if (!p64.empty()) memcpy(blob.data() + h.off64, p64.data(), p64.size() * 8);
    if (!p32.empty()) memcpy(blob.data() + h.off32, p32.data(), p32.size() * 4);
    if (!p16.empty()) memcpy(blob.data() + h.off16, p16.data(), p16.size() * 2);
    if (!p8.empty()) memcpy(blob.data() + h.off8, p8.data(), p8.size());
    fgpu_set_blob(ctx, blob.data(), blob.size());
}

bool facul_gpu::fits(cxx_mpz const & n)
{
    return mpz_odd_p(n) && mpz_sizeinbase(n, 2) <= 126 && mpz_cmp_ui(n, 1) > 0;
}

fgpu_job facul_gpu::make_job(cxx_mpz const & n, int chain)
{
    fgpu_job j {};
    j.n[0] = mpz_getlimbn_uint64(n, 0);
    j.n[1] = mpz_sizeinbase(n, 2) > 64 ? mpz_getlimbn_uint64(n, 1) : 0;
    j.chain = chain;
    return j;
}

cxx_mpz facul_gpu::prime(fgpu_result const & r, int i)
{
    cxx_mpz z;
    mpz_set_uint64(z, r.p[i][1]);
    mpz_mul_2exp(z, z, 64);
    cxx_mpz lo;
    mpz_set_uint64(lo, r.p[i][0]);
    mpz_add(z, z, lo);
    return z;
}

void facul_gpu::run(std::vector<fgpu_job> const & jobs, std::vector<fgpu_result> & res)
{
    res.assign(jobs.size(), fgpu_result {});
    std::lock_guard<std::mutex> const dummy(run_lock);
    {
        std::lock_guard<std::mutex> const d2(lock);
        if (dirty) {
            upload();
            dirty = false;
        }
    }
    /* by word count */
    std::vector<size_t> idx[2];
    for (size_t i = 0; i < jobs.size(); i++) {
        auto const & j = jobs[i];
        if (!(j.n[0] & 1) || (j.n[1] >> 62)) {
            res[i].status = FGPU_CPU;
            continue;
        }
        idx[j.n[1] ? 1 : 0].push_back(i);
    }
    std::vector<fgpu_job> sub;
    std::vector<fgpu_result> out;
    for (int w = 0; w < 2; w++) {
        if (idx[w].empty()) continue;
        /* sort by chain so that a warp runs one chain */
        std::stable_sort(idx[w].begin(), idx[w].end(),
                [&](size_t a, size_t b) { return jobs[a].chain < jobs[b].chain; });
        sub.clear();
        for (size_t i : idx[w]) sub.push_back(jobs[i]);
        out.resize(sub.size());
        fgpu_run(ctx, sub.data(), out.data(), sub.size(), w + 1);
        for (size_t k = 0; k < idx[w].size(); k++)
            res[idx[w][k]] = out[k];
    }
}


void facul_gpu::factor_all(std::vector<std::vector<cxx_mpz> const *> const & N,
        facul_strategies const & strat, std::vector<facul_gpu_outcome> & out)
{
    size_t const nc = N.size();
    out.assign(nc, facul_gpu_outcome {});

    /* per candidate: the sides that need methods, in the order of the
     * strategy (the CPU starts with the side of methods[0]) */
    struct pending { int side[2]; int chain[2]; int n; int next; };
    std::vector<pending> pend(nc);

    for (size_t c = 0; c < nc; c++) {
        auto const & n = *N[c];
        auto & o = out[c];
        size_t const nsides = n.size();
        if (nsides != 2 || strat.B.size() != 2) { o.cpu = true; continue; }
        o.primes.assign(nsides, {});
        o.status = FACUL_SMOOTH;
        std::vector<unsigned int> sizes(nsides);
        for (size_t s = 0; s < nsides; s++) sizes[s] = mpz_sizeinbase(n[s], 2);
        auto const & meths = strat(sizes);
        pending & P = pend[c];
        P.n = 0;
        P.next = 0;
        bool need[2] = { false, false };
        for (size_t s = 0; s < nsides && !o.cpu; s++) {
            /* the quick tests of facul_all */
            if (mpz_cmp_ui(n[s], 1) == 0) continue;
            if (mpz_get_d(n[s]) < strat.BB[s]) {
                o.primes[s].emplace_back(n[s]);
                continue;
            } else if (strat.BB[s] == 0) {
                /* not perfectly sieved (this is what las uses: see
                 * facul_make_strategies): facul_all's test, on the host */
                auto const Bd = double(strat.B[s]);
                if (mpz_get_d(n[s]) < Bd * Bd && mpz_probab_prime_p(n[s], 1)) {
                    o.primes[s].emplace_back(n[s]);
                    continue;
                }
            }
            if (mpz_sgn(n[s]) < 0 || !fits(n[s])) {
                o.cpu = true;
                break;
            }
            need[s] = true;
        }
        if (o.cpu) continue;
        if (meths.empty() && (need[0] || need[1])) { o.cpu = true; continue; }
        for (auto const & m : meths) {
            int const s = m.side;
            if (s < 0 || s >= 2 || !need[s]) continue;
            need[s] = false;
            int const id = chain(meths, s, strat.lpb[s], strat.BB[s], strat.BBB[s]);
            if (id < 0) { o.cpu = true; break; }
            P.side[P.n] = s;
            P.chain[P.n] = id;
            P.n++;
        }
        /* a side with work but no method: facul_all leaves it not smooth */
        if (!o.cpu && (need[0] || need[1])) o.status = FACUL_NOT_SMOOTH;
    }

    std::vector<fgpu_job> jobs;
    std::vector<size_t> who;
    std::vector<fgpu_result> res;
    for (int round = 0; round < 2; round++) {
        jobs.clear();
        who.clear();
        for (size_t c = 0; c < nc; c++) {
            auto & o = out[c];
            pending & P = pend[c];
            if (o.cpu || o.status != FACUL_SMOOTH || P.next >= P.n) continue;
            int const s = P.side[P.next];
            jobs.push_back(make_job((*N[c])[s], P.chain[P.next]));
            who.push_back(c);
        }
        if (jobs.empty()) break;
        run(jobs, res);
        for (size_t k = 0; k < jobs.size(); k++) {
            size_t const c = who[k];
            auto & o = out[c];
            pending & P = pend[c];
            int const s = P.side[P.next++];
            fgpu_result const & r = res[k];
            switch (r.status) {
                case FGPU_SMOOTH:
                    for (int i = 0; i < r.nprimes; i++)
                        o.primes[s].push_back(prime(r, i));
                    break;
                case FGPU_NOT_SMOOTH: o.status = FACUL_NOT_SMOOTH; break;
                case FGPU_MAYBE: o.status = FACUL_MAYBE; break;
                default: o.cpu = true;
            }
        }
    }

    /* as facul_all: sorted primes; and check them */
    for (size_t c = 0; c < nc; c++) {
        auto & o = out[c];
        if (o.cpu || o.status != FACUL_SMOOTH) continue;
        for (size_t s = 0; s < o.primes.size(); s++) {
            std::sort(o.primes[s].begin(), o.primes[s].end());
            cxx_mpz z = 1;
            for (auto const & p : o.primes[s]) mpz_mul(z, z, p);
            if (mpz_cmp(z, (*N[c])[s]) != 0) {
                fprintf(stderr, "facul-gpu: wrong factorization of a"
                        " cofactor, redone on the CPU\n");
                o.cpu = true;
            }
        }
    }
}

#else /* HAVE_CUDA */

facul_gpu * facul_gpu::get() { return nullptr; }
facul_gpu::~facul_gpu() = default;
int facul_gpu::chain(std::vector<facul_method_side> const &, int,
        unsigned long, double, double) { return -1; }
void facul_gpu::run(std::vector<fgpu_job> const & jobs, std::vector<fgpu_result> & res)
{
    res.assign(jobs.size(), fgpu_result {});
    for (auto & r : res) r.status = FGPU_CPU;
}
bool facul_gpu::fits(cxx_mpz const &) { return false; }
fgpu_job facul_gpu::make_job(cxx_mpz const &, int) { return {}; }
cxx_mpz facul_gpu::prime(fgpu_result const &, int) { return {}; }
int facul_gpu::device() const { return -1; }
void facul_gpu::stats(double & k, unsigned long & n) const { k = 0; n = 0; }
void facul_gpu::factor_all(std::vector<std::vector<cxx_mpz> const *> const & N,
        facul_strategies const &, std::vector<facul_gpu_outcome> & out)
{
    out.assign(N.size(), facul_gpu_outcome {});
    for (auto & o : out) o.cpu = true;
}

#endif /* HAVE_CUDA */

/* vim: set sw=4 fdm=marker: */
