/* GPU size optimization; see sopt_gpu_kernels.h.
 *
 * Built with -fmad=false: every a*b+c below is two roundings, as on the
 * CPU (whose build has no FMA contraction); explicit fma() calls are the
 * CPU's explicit fma() calls. The functions follow, operation for
 * operation, polynomial.hpp (eval, bound_positive_roots,
 * positive_roots_from_derivative_sign_changes, findroot_falseposition,
 * findroot_dichotomy, derivative) and polyselect_norms.cpp (L2_lognorm,
 * L2_skewness_derivative_numerator, L2_skewness, L2_skew_lognorm).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

#include <cuda_runtime.h>
#include "cado_cuda_check.cuh"
#include "gpu_claim.h"

#include "sopt_gpu_kernels.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {           \
    fprintf(stderr, "sopt GPU: %s failed at %s:%d: %s\n", #x,               \
            __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)

#define MAXD 8          /* polynomial degrees handled */

/* the table of polyselect_norms.cpp, rows 0 .. MAXD */
__constant__ uint32_t coeffs_integral[MAXD + 1][MAXD / 2 + 1] = {
    { 1 },
    { 2 },
    { 1, 6 },
    { 4, 20 },
    { 3, 10, 70 },
    { 12, 28, 252 },
    { 10, 28, 84, 924 },
    { 40, 72, 264, 3432 },
    { 35, 90, 198, 858, 12870 },
};

/* ------------------------------------------------------------------ */
/* fixed-width integers                                                */

template<int NL> struct bn { uint64_t w[NL]; };

template<int NL> __device__ __forceinline__ bool bn_neg_p(bn<NL> const & x)
{
    return (int64_t) x.w[NL - 1] < 0;
}

template<int NL> __device__ __forceinline__ bool bn_zero_p(bn<NL> const & x)
{
    uint64_t o = 0;
    for (int i = 0; i < NL; i++) o |= x.w[i];
    return o == 0;
}

template<int NL> __device__ __forceinline__ bn<NL> bn_negate(bn<NL> const & x)
{
    bn<NL> r;
    unsigned int c = 1;
    for (int i = 0; i < NL; i++) {
        uint64_t v = ~x.w[i];
        r.w[i] = v + c;
        c = c && r.w[i] == 0;
    }
    return r;
}

template<int NL> __device__ __forceinline__ bn<NL> bn_abs(bn<NL> const & x)
{
    return bn_neg_p(x) ? bn_negate(x) : x;
}

template<int NL> __device__ __forceinline__ bn<NL> bn_add(bn<NL> const & a, bn<NL> const & b)
{
    bn<NL> r;
    uint64_t c = 0;
    for (int i = 0; i < NL; i++) {
        uint64_t s = a.w[i] + c;
        uint64_t c1 = s < c;
        r.w[i] = s + b.w[i];
        c = c1 + (r.w[i] < s);
    }
    return r;
}

/* |x| < 2^(64 NL - 3) */
template<int NL> __device__ __forceinline__ bool bn_fits(bn<NL> const & x)
{
    int64_t t = (int64_t) x.w[NL - 1] >> 60;
    return t == 0 || t == -1;
}

/* bit length of |x|, as mpz_sizeinbase(x, 2) (1 for 0) */
template<int NL> __device__ int bn_sizeinbase2(bn<NL> const & x)
{
    bn<NL> a = bn_abs(x);
    for (int i = NL; i--; )
        if (a.w[i])
            return 64 * i + 64 - __clzll(a.w[i]);
    return 1;
}

/* x * 2^e, e >= 0; the caller checks that it fits */
template<int NL> __device__ bn<NL> bn_shl(bn<NL> const & x, int e)
{
    bn<NL> r;
    const int q = e / 64, s = e % 64;
    for (int i = NL; i--; ) {
        uint64_t hi = i - q >= 0 ? x.w[i - q] : 0;
        uint64_t lo = i - q - 1 >= 0 ? x.w[i - q - 1] : 0;
        r.w[i] = s ? (hi << s) | (lo >> (64 - s)) : hi;
    }
    return r;
}

/* x + sigma * 2^e * y (sigma = +-1); false if it does not fit */
template<int NL> __device__ bool bn_addmul_2exp(bn<NL> & x, bn<NL> const & y,
        int sigma, int e)
{
    if (bn_zero_p(y)) return true;
    if (bn_sizeinbase2(y) + e >= 64 * NL - 4) return false;
    bn<NL> t = bn_shl(y, e);
    if (sigma < 0) t = bn_negate(t);
    x = bn_add(x, t);
    return bn_fits(x);
}

/* mpz_get_d_2exp then ldexp(d, e + scale) */
template<int NL> __device__ double bn_get_d_scaled(bn<NL> const & x, int scale)
{
    if (bn_zero_p(x)) return ldexp(0.0, scale);
    const bool neg = bn_neg_p(x);
    bn<NL> a = bn_abs(x);
    const int L = bn_sizeinbase2(a);
    double d;
    if (L <= 53) {
        d = ldexp((double) a.w[0], scale);
    } else {
        /* the top 53 bits, truncated */
        const int sh = L - 53;
        const int q = sh / 64, s = sh % 64;
        uint64_t lo = a.w[q], hi = q + 1 < NL ? a.w[q + 1] : 0;
        uint64_t m = s ? (lo >> s) | (hi << (64 - s)) : lo;
        m &= (1ull << 53) - 1;
        d = ldexp((double) m, sh + scale);
    }
    return neg ? -d : d;
}

/* ------------------------------------------------------------------ */
/* polynomials with double coefficients                                */

struct dpoly {
    int deg;
    double c[MAXD + 1];
};

__device__ __forceinline__ int sgn(double v) { return (0.0 < v) - (v < 0.0); }

/* Horner with fma, as eval_with_reference */
__device__ double dp_eval(dpoly const & p, double x)
{
    if (p.deg < 0) return 0;
    double s = p.c[p.deg];
    for (int k = p.deg; k--; )
        s = fma(s, x, p.c[k]);
    return s;
}

__device__ void dp_derivative(dpoly & r, dpoly const & p)
{
    if (p.deg <= 0) { r.deg = -1; return; }
    r.deg = p.deg - 1;
    for (int i = 1; i <= p.deg; i++)
        r.c[i - 1] = p.c[i] * i;
}

__device__ double dp_bound_positive_roots(dpoly const & p)
{
    const int d = p.deg;
    const int s = p.c[d] < 0;
    dpoly q;
    q.deg = d;
    for (int i = 0; i <= d; i++) q.c[i] = 0;
    for (int i = 0; i < d; i++) {
        double v = s ? -p.c[i] : p.c[i];
        if (v < 0) q.c[i] = v;
    }
    q.c[d] = fabs(p.c[d]);
    double b = 1;
    for ( ; dp_eval(q, b) <= 0 ; b = b + b) ;
    return b;
}

__device__ double dp_findroot_dichotomy(dpoly const & p, double a, double b, double sa)
{
    double s;
    for (;;) {
        s = (a + b) / 2;
        if (s == a || s == b) return s;
        if (dp_eval(p, s) * sa > 0)
            a = s;
        else
            b = s;
    }
}

__device__ double dp_findroot_falseposition(dpoly const & p, double a0, double b0,
        double pa0, double pb0)
{
    int side = 0;
    double a = a0, b = b0;
    if (a == b) return a;
    const int sigma = sgn(b - a);
    double pa = pa0, pb = pb0;
    for (;;) {
        double s = (a * pb - b * pa) / (pb - pa);
        double middle = (a + b) / 2;
        const bool escapes_range = sigma * s < sigma * a || sigma * s > sigma * b;
        const bool hits_bounds = s == a || s == b;
        const bool middle_cut_is_nice = !(middle == a || middle == b);
        if (escapes_range || (hits_bounds && middle_cut_is_nice))
            s = middle;
        if (s == a || s == b) return s;
        double ps = dp_eval(p, s);
        if (ps * pa > 0) {
            a = s; pa = ps;
            if (side == 1) pb /= 2;
            side = 1;
        } else {
            b = s; pb = ps;
            if (side == -1) pa /= 2;
            side = -1;
        }
        if (isnan(b))
            return dp_findroot_dichotomy(p, a0, b0, pa0);
    }
}

/* v (nv entries) holds the positive roots of p's derivative; replaced by
 * the positive roots of p */
__device__ void dp_roots_from_derivative(dpoly const & p, double * v, int & nv,
        double bound)
{
    if (p.deg == 1) {
        if (p.c[0] != 0 && p.c[0] * dp_eval(p, bound) <= 0) {
            v[0] = -p.c[0] / p.c[1];
            nv = 1;
        }
        return;
    }
    double a = 0;
    double va = p.c[0];
    v[nv++] = bound;
    int m = 0;
    int k = 1;
    for ( ; k <= p.deg && !p.c[k] ; k++) ;
    const int c01 = sgn(p.c[0]) * sgn(p.c[k]) * ((k & 1) ? 1 : sgn(bound));
    bool no_chance = c01 * sgn(bound) > 0 || p.c[0] == 0;
    const int n = nv;
    for (int i = 0; i < n; i++) {
        double b = v[i];
        double vb = dp_eval(p, b);
        if (no_chance) {
            no_chance = false;
        } else if (va * vb < 0) {
            v[m++] = dp_findroot_falseposition(p, a, b, va, vb);
        } else {
            no_chance = true;
        }
        a = b;
        va = vb;
    }
    nv = m;
}

/* positive_roots(): roots in v, their number returned */
__device__ int dp_positive_roots(dpoly const & p, double * v)
{
    const int d = p.deg;
    if (d <= 0) return 0;
    const double B = dp_bound_positive_roots(p);
    dpoly dg[MAXD];
    dg[0] = p;
    for (int k = 1; k < d; k++)
        dp_derivative(dg[k], dg[k - 1]);
    int nv = 0;
    for (int k = d; k--; )
        dp_roots_from_derivative(dg[k], v, nv, B);
    return nv;
}

/* the argument of the log in L2_lognorm (before the log) */
__device__ double dp_L2_tt(dpoly const & p, double s)
{
    const int d = p.deg;
    const double * a = p.c;
    const double invs = 1.0 / s;
    const double s2 = s * s;
    const double is2 = invs * invs;
    double tt = 0;
    double u0 = (d & 1) ? invs : 1.0;
    double u1 = (d & 1) ? s : 1.0;
    const int p0 = d / 2;
    const int p1 = d - p0;
    for (int k = 0; k <= d / 2; k++) {
        double t0 = 0;
        double t1 = 0;
        for (int i = 1; i <= d / 2 - k; i++) {
            t0 += a[p0 - k - i] * a[p0 - k + i];
            t1 += a[p1 + k - i] * a[p1 + k + i];
        }
        t0 = (2 * t0 + a[p0 - k] * a[p0 - k]) * u0;
        t1 = (2 * t1 + a[p1 + k] * a[p1 + k]) * u1;
        u0 *= is2;
        u1 *= s2;
        tt += (double) coeffs_integral[d][k] * (t0 + t1);
    }
    return ldexp(tt * 0x1.921fb54442d18p+1 / (d + 1), -2 * d);
}

/* L2_skewness_derivative_numerator; false on inf / nan */
__device__ bool dp_L2_dnum(dpoly & dP, dpoly const & p)
{
    const int d = p.deg;
    const double * a = p.c;
    for (int k = 0; k <= d; k++) {
        double t0 = 0;
        for (int i = 1; i <= k && i <= d - k; i++)
            t0 += a[k - i] * a[k + i];
        t0 = (2 * t0 + a[k] * a[k]);
        const int j = (abs(d - 2 * k) - (d & 1)) / 2;
        t0 = t0 * coeffs_integral[d][j] * (2 * k - d);
        dP.c[k] = t0;
    }
    /* dP[k] = t0 strips zero coefficients at the top */
    int dd = d;
    for ( ; dd >= 0 && dP.c[dd] == 0 ; dd--) ;
    dP.deg = dd;
    for (int k = 0; k <= dd; k++)
        if (isinf(dP.c[k]) || isnan(dP.c[k])) return false;
    return true;
}

/* unit in the last place of v */
__device__ __forceinline__ double ulp(double v)
{
    return v == 0 ? 0x1p-1074 : ldexp(1.0, ilogb(v) - 52);
}

/* A GPU value x with error bound ex: the CPU's value is within ex of it.
 * The only difference is the log: glibc's is within 0.52 ulp of the true
 * value, CUDA's within 1 ulp, so they differ by less than 2 ulp; what is
 * computed from the log afterwards (/2, an fma, an addition) adds at
 * most one rounding each, counted as 2 ulp of the result. A decision
 * x < y is the CPU's when |x - y| > ex + ey; otherwise it is left to the
 * CPU. */
__device__ __forceinline__ bool too_close(double x, double ex, double y, double ey)
{
    return fabs(x - y) <= ex + ey;
}

/* L2_skew_lognorm(f), approximately (GPU log); status as in the header.
 * If tt != NULL, the exact argument of the log and fix are returned. */
template<int NL>
__device__ double L2_skew_lognorm_gpu(bn<NL> const * f, int deg, int & status,
        double & err, double * skew_out = nullptr, double * tt_out = nullptr,
        int * fix_out = nullptr)
{
    const int bits0 = bn_sizeinbase2(f[0]);
    const int bits1 = bn_sizeinbase2(f[deg]);
    const int scale = (bits1 - bits0) / deg;
    dpoly F;
    F.deg = deg;
    for (int i = 0; i <= deg; i++)
        F.c[i] = bn_get_d_scaled(f[i], -bits0 + -scale * i);

    /* L2_skewness(F) */
    dpoly dP;
    if (!dp_L2_dnum(dP, F)) { status = SSG_NONFINITE; return 0; }
    double roots[MAXD + 2];
    const int nr = dp_positive_roots(dP, roots);
    if (nr == 0) { status = SSG_NOROOT; return 0; }
    double best_s = sqrt(roots[0]);
    double tt = dp_L2_tt(F, best_s);
    if (isnan(tt) || isinf(tt)) { status = SSG_NONFINITE; return 0; }
    double logtt = log(tt);
    double l = logtt / 2;
    for (int i = 1; i < nr; i++) {
        const double s = sqrt(roots[i]);
        const double tti = dp_L2_tt(F, s);
        if (isnan(tti) || isinf(tti)) { status = SSG_NONFINITE; return 0; }
        const double logtti = log(tti);
        const double li = logtti / 2;
        if (too_close(li, 2 * ulp(logtti), l, 2 * ulp(logtt))) { status = SSG_CLOSE; return 0; }
        if (li < l) {
            best_s = s;
            l = li;
            tt = tti;
            logtt = logtti;
        }
    }
    const int fix = 2 * bits0 + deg * scale;
    if (skew_out) *skew_out = ldexp(best_s, -scale);
    if (tt_out) *tt_out = tt;
    if (fix_out) *fix_out = fix;
    const double r = fma((double) fix, 0x1.62e42fefa39efp-2, l);
    err = 2 * ulp(logtt) + 2 * ulp(r);
    return r;
}

template<int NL>
__global__ void l2_test_kernel(int n, int deg, const uint64_t * coeffs,
        double * skew, double * tt, int * fix, double * approx, int * status)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    bn<NL> f[MAXD + 1];
    for (int i = 0; i <= deg; i++)
        for (int l = 0; l < NL; l++)
            f[i].w[l] = coeffs[((size_t) t * (deg + 1) + i) * NL + l];
    int st = SSG_OK;
    double err;
    approx[t] = L2_skew_lognorm_gpu<NL>(f, deg, st, err, skew + t, tt + t, fix + t);
    status[t] = st;
}

/* ------------------------------------------------------------------ */
/* the local descent                                                   */

/* f(x + sigma 2^e) */
template<int NL> __device__ bool bn_translation(bn<NL> * ft, bn<NL> const * f, int d,
        int sigma, int e)
{
    for (int i = 0; i <= d; i++) ft[i] = f[i];
    for (int i = d - 1; i >= 0; i--)
        for (int j = i; j < d; j++)
            if (!bn_addmul_2exp(ft[j], ft[j + 1], sigma, e)) return false;
    return true;
}

/* f + sigma 2^e x^t g, g linear */
template<int NL> __device__ bool bn_rotation(bn<NL> * fr, bn<NL> const * f, int d,
        bn<NL> const * g, int sigma, int e, int t)
{
    for (int i = 0; i <= d; i++) fr[i] = f[i];
    return bn_addmul_2exp(fr[t], g[0], sigma, e)
        && bn_addmul_2exp(fr[t + 1], g[1], sigma, e);
}

#define GUARD 0.001     /* SOPT_LOCAL_DESCENT_GUARD */
#define MAXROT 7        /* SOPT_MAX_DEGREE_ROTATION */

/* fail with this status (a CPU descent follows) */
#define FAIL(s) do { status[t] = (s); return; } while (0)
/* lm = L2_skew_lognorm(poly) with its error bound e, or fail */
#define L2(lm, e, poly) do { int st_ = SSG_OK;                           \
        lm = L2_skew_lognorm_gpu<NL>(poly, d, st_, e);                  \
        if (st_ != SSG_OK) FAIL(st_); } while (0)
/* r = (x > y), or r = (x < y), or fail if too close to tell */
#define GT(r, x, ex, y, ey) do { const double x_ = (x), y_ = (y);       \
        if (too_close(x_, (ex), y_, (ey))) FAIL(SSG_CLOSE); r = x_ > y_; } while (0)
#define LT(r, x, ex, y, ey) do { const double x_ = (x), y_ = (y);       \
        if (too_close(x_, (ex), y_, (ey))) FAIL(SSG_CLOSE); r = x_ < y_; } while (0)
/* logmu_opt + GUARD, rounded: its error bound */
#define EGUARD (e_opt + 2 * ulp(logmu_opt + GUARD))

template<int NL>
__global__ void descent_kernel(int n, int d, uint64_t * fio, uint64_t * gio,
        int deg_rotation, unsigned int max_iter, int * status, int * state)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    bn<NL> f_raw[MAXD + 1], g_raw[2], f_opt[MAXD + 1], g_opt[2], ftmp[MAXD + 1];
    for (int i = 0; i <= d; i++)
        for (int l = 0; l < NL; l++)
            f_raw[i].w[l] = fio[((size_t) t * (d + 1) + i) * NL + l];
    for (int i = 0; i < 2; i++)
        for (int l = 0; l < NL; l++)
            g_raw[i].w[l] = gio[((size_t) t * 2 + i) * NL + l];

    double logmu_opt, logmu, e_opt, e;
    bool c;
    L2(logmu_opt, e_opt, f_raw);

    /* k[i] = 2^ek[i], kt = 2^ekt */
    int ek[MAXROT + 1], ekt;
    for (int i = 0; i <= deg_rotation; i++) {
        ek[i] = 0;
        for (;;) {
            if (!bn_rotation<NL>(ftmp, f_raw, d, g_raw, 1, ek[i], i)) FAIL(SSG_OVERFLOW);
            L2(logmu, e, ftmp);
            GT(c, logmu, e, logmu_opt + GUARD, EGUARD);
            if (c) {
                if (!bn_rotation<NL>(ftmp, f_raw, d, g_raw, -1, ek[i], i)) FAIL(SSG_OVERFLOW);
                L2(logmu, e, ftmp);
                GT(c, logmu, e, logmu_opt + GUARD, EGUARD);
                if (c)
                    break;
            }
            ek[i]++;
        }
    }
    ekt = 0;
    for (;;) {
        if (!bn_translation<NL>(ftmp, f_raw, d, 1, ekt)) FAIL(SSG_OVERFLOW);
        L2(logmu, e, ftmp);
        GT(c, logmu, e, logmu_opt + GUARD, EGUARD);
        if (c) {
            if (!bn_translation<NL>(ftmp, f_raw, d, -1, ekt)) FAIL(SSG_OVERFLOW);
            L2(logmu, e, ftmp);
            GT(c, logmu, e, logmu_opt + GUARD, EGUARD);
            if (c)
                break;
        }
        ekt++;
    }

    for (int i = 0; i <= d; i++) f_opt[i] = f_raw[i];
    g_opt[0] = g_raw[0];
    g_opt[1] = g_raw[1];

    unsigned int iter = 0;
    /* in the main loop, a close comparison returns the state at the
       beginning of the iteration (the CPU continues from there) */
    bn<NL> f_save[MAXD + 1], g_save[2];
    unsigned int iter_save = 0;
    int ekt_save = 0;
    bool in_loop = false;
    while (iter < max_iter) {
        for (int i = 0; i <= d; i++) f_save[i] = f_opt[i];
        g_save[0] = g_opt[0];
        g_save[1] = g_opt[1];
        iter_save = iter;
        ekt_save = ekt;
        int ek_save[MAXROT + 1];
        for (int i = 0; i <= deg_rotation; i++) ek_save[i] = ek[i];
        in_loop = true;
        (void) in_loop;
#undef FAIL
#define FAIL(s) do {                                                    \
            if ((s) == SSG_CLOSE) {                                     \
                for (int i_ = 0; i_ <= d; i_++)                         \
                    for (int l_ = 0; l_ < NL; l_++)                     \
                        fio[((size_t) t * (d + 1) + i_) * NL + l_] = f_save[i_].w[l_]; \
                for (int i_ = 0; i_ < 2; i_++)                          \
                    for (int l_ = 0; l_ < NL; l_++)                     \
                        gio[((size_t) t * 2 + i_) * NL + l_] = g_save[i_].w[l_]; \
                int * st_ = state + (size_t) t * SSG_STATE_INTS;        \
                st_[0] = (int) iter_save;                               \
                st_[1] = ekt_save;                                      \
                for (int i_ = 0; i_ <= deg_rotation; i_++) st_[2 + i_] = ek_save[i_]; \
                status[t] = SSG_RESUME;                                 \
            } else                                                      \
                status[t] = (s);                                        \
            return; } while (0)
        iter++;
        int changedt = 0;
        int changed[MAXROT + 1];
        for (int i = 0; i <= deg_rotation; i++) changed[i] = 0;

        /* translation by kt, then -kt */
        if (!bn_translation<NL>(ftmp, f_opt, d, 1, ekt)) FAIL(SSG_OVERFLOW);
        L2(logmu, e, ftmp);
        LT(c, logmu, e, logmu_opt, e_opt);
        if (c) {
            changedt = 1;
            logmu_opt = logmu;
            e_opt = e;
            for (int i = 0; i <= d; i++) f_opt[i] = ftmp[i];
            /* g(x + kt) */
            if (!bn_addmul_2exp(g_opt[0], g_opt[1], 1, ekt)) FAIL(SSG_OVERFLOW);
        } else {
            if (!bn_translation<NL>(ftmp, f_opt, d, -1, ekt)) FAIL(SSG_OVERFLOW);
            L2(logmu, e, ftmp);
            LT(c, logmu, e, logmu_opt, e_opt);
            if (c) {
                changedt = 1;
                logmu_opt = logmu;
                e_opt = e;
            e_opt = e;
                for (int i = 0; i <= d; i++) f_opt[i] = ftmp[i];
                if (!bn_addmul_2exp(g_opt[0], g_opt[1], -1, ekt)) FAIL(SSG_OVERFLOW);
            }
        }

        for (int i = 0; i <= deg_rotation; i++) {
            if (!bn_rotation<NL>(ftmp, f_opt, d, g_opt, 1, ek[i], i)) FAIL(SSG_OVERFLOW);
            L2(logmu, e, ftmp);
            LT(c, logmu, e, logmu_opt, e_opt);
            if (c) {
                changed[i] = 1;
                logmu_opt = logmu;
                e_opt = e;
            e_opt = e;
                for (int j = 0; j <= d; j++) f_opt[j] = ftmp[j];
            } else {
                if (!bn_rotation<NL>(ftmp, f_opt, d, g_opt, -1, ek[i], i)) FAIL(SSG_OVERFLOW);
                L2(logmu, e, ftmp);
                LT(c, logmu, e, logmu_opt, e_opt);
                if (c) {
                    changed[i] = 1;
                    logmu_opt = logmu;
                    e_opt = e;
                e_opt = e;
            e_opt = e;
                    for (int j = 0; j <= d; j++) f_opt[j] = ftmp[j];
                }
            }
        }

        bool is_finished = changedt == 0 && ekt == 0;
        for (int i = 0; i <= deg_rotation && is_finished; i++)
            is_finished = changed[i] == 0 && ek[i] == 0;
        if (is_finished)
            break;

        if (changedt == 1)
            ekt++;
        else if (ekt > 0)
            ekt--;
        for (int i = 0; i <= deg_rotation; i++) {
            if (changed[i] == 1)
                ek[i]++;
            else if (ek[i] > 0)
                ek[i]--;
        }
    }
#undef FAIL
#define FAIL(s) do { status[t] = (s); return; } while (0)

    for (int i = 0; i <= d; i++)
        for (int l = 0; l < NL; l++)
            fio[((size_t) t * (d + 1) + i) * NL + l] = f_opt[i].w[l];
    for (int i = 0; i < 2; i++)
        for (int l = 0; l < NL; l++)
            gio[((size_t) t * 2 + i) * NL + l] = g_opt[i].w[l];
    status[t] = SSG_OK;
}

/* ------------------------------------------------------------------ */
/* host                                                                */

struct ssg_ctx {
    int device = 0;
    cudaStream_t stream = nullptr;
    void * d_buf = nullptr; size_t cap = 0;
    double seconds = 0;
    unsigned long descents = 0, returned = 0;
};

static double now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

struct ssg_ctx * ssg_ctx_create(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    auto * c = new ssg_ctx;
    const char * e = getenv("CADO_POLYSELECT_GPU_DEVICE");
    if (!e) e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!e) e = getenv("SLURM_LOCALID");
    c->device = (e ? atoi(e) : cado_gpu_claim("polyselect", ndev)) % ndev;
    CK(cudaSetDevice(c->device));
    if (!cado_cuda_can_run("polyselect GPU size optimization")) { delete c; return nullptr; }
    CK(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    return c;
}

void ssg_ctx_free(struct ssg_ctx * c)
{
    if (!c) return;
    cudaSetDevice(c->device);
    cudaFree(c->d_buf);
    cudaStreamDestroy(c->stream);
    delete c;
}

static void * scratch(struct ssg_ctx * c, size_t bytes)
{
    if (bytes > c->cap) {
        cudaFree(c->d_buf);
        c->cap = bytes + bytes / 2 + 4096;
        CK(cudaMalloc(&c->d_buf, c->cap));
    }
    return c->d_buf;
}

#define DISPATCH_NL(nl, CALL) do {                                     \
    switch (nl) {                                                       \
        case 2: { constexpr int NL = 2; CALL; } break;                  \
        case 4: { constexpr int NL = 4; CALL; } break;                  \
        case 8: { constexpr int NL = 8; CALL; } break;                  \
        case 16: { constexpr int NL = 16; CALL; } break;                \
        default: abort();                                               \
    } } while (0)

void ssg_l2_test(struct ssg_ctx * c, int n, int deg, int nl,
        const uint64_t * coeffs, double * skew, double * tt, int * fix,
        double * approx, int * status)
{
    CK(cudaSetDevice(c->device));
    const size_t cb = (size_t) n * (deg + 1) * nl * sizeof(uint64_t);
    char * p = (char *) scratch(c, cb + (size_t) n * (3 * sizeof(double) + 2 * sizeof(int)) + 64);
    uint64_t * d_c = (uint64_t *) p;
    double * d_skew = (double *) (p + ((cb + 7) & ~7));
    double * d_tt = d_skew + n;
    double * d_ap = d_tt + n;
    int * d_fix = (int *) (d_ap + n);
    int * d_st = d_fix + n;
    CK(cudaMemcpyAsync(d_c, coeffs, cb, cudaMemcpyHostToDevice, c->stream));
    const int T = 128;
    DISPATCH_NL(nl, (l2_test_kernel<NL><<<(n + T - 1) / T, T, 0, c->stream>>>(
                    n, deg, d_c, d_skew, d_tt, d_fix, d_ap, d_st)));
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(skew, d_skew, n * sizeof(double), cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(tt, d_tt, n * sizeof(double), cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(approx, d_ap, n * sizeof(double), cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(fix, d_fix, n * sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(status, d_st, n * sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
}

void ssg_descent(struct ssg_ctx * c, int n, int deg, int nl, uint64_t * f,
        uint64_t * g, int deg_rotation, unsigned int max_iter, int * status,
        int * state)
{
    if (n == 0) return;
    const double t0 = now();
    CK(cudaSetDevice(c->device));
    const size_t fb = (size_t) n * (deg + 1) * nl * sizeof(uint64_t);
    const size_t gb = (size_t) n * 2 * nl * sizeof(uint64_t);
    char * p = (char *) scratch(c, fb + gb + (size_t) n * (1 + SSG_STATE_INTS) * sizeof(int) + 64);
    uint64_t * d_f = (uint64_t *) p;
    uint64_t * d_g = (uint64_t *) (p + fb);
    int * d_st = (int *) (p + fb + gb);
    int * d_state = d_st + n;
    CK(cudaMemcpyAsync(d_f, f, fb, cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyAsync(d_g, g, gb, cudaMemcpyHostToDevice, c->stream));
    const int T = 64;
    DISPATCH_NL(nl, (descent_kernel<NL><<<(n + T - 1) / T, T, 0, c->stream>>>(
                    n, deg, d_f, d_g, deg_rotation, max_iter, d_st, d_state)));
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(f, d_f, fb, cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(g, d_g, gb, cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(status, d_st, n * sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CK(cudaMemcpyAsync(state, d_state, (size_t) n * SSG_STATE_INTS * sizeof(int),
                cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    c->descents += n;
    for (int i = 0; i < n; i++) c->returned += status[i] != SSG_OK;
    c->seconds += now() - t0;
}

int ssg_fp64_ratio(const struct ssg_ctx * c)
{
    int r = 0;
    if (cudaDeviceGetAttribute(&r, cudaDevAttrSingleToDoublePrecisionPerfRatio,
                c->device) != cudaSuccess)
        return 0;
    return r;
}

void ssg_ctx_stats(const struct ssg_ctx * c, double * seconds,
        unsigned long * descents, unsigned long * returned)
{
    *seconds = c->seconds;
    *descents = c->descents;
    *returned = c->returned;
}
