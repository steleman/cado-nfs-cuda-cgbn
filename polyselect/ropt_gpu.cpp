#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "macros.h"
#include "gmp_aux.h"
#include "polyselect_alpha.h"
#include "rootfinder.h"
#include "ropt_gpu.hpp"

#ifdef HAVE_CUDA
#include "ropt_gpu_kernels.h"

struct ropt_sieve_oplist {
  struct rsg_ctx * ctx = nullptr;
  struct rsg_op * buf = nullptr;
  size_t n = 0, cap = 0;
};

namespace {
struct ropt_alpha_batch {
  unsigned long B = 0;
  std::vector<uint32_t> primes;          /* the primes <= B */
  std::vector<cxx_mpz_poly> fs;
  std::vector<unsigned short> counts;    /* fs.size() x primes.size() */
};

struct ropt_gpu_holder {
  ropt_sieve_oplist ol;
  ropt_alpha_batch alpha;
  bool tried = false;
  ~ropt_gpu_holder () {
    if (!ol.ctx)
      return;
    if (getenv ("CADO_ROPT_GPU_VERBOSE")) {
      double sec;
      unsigned long calls, ops;
      rsg_ctx_stats (ol.ctx, &sec, &calls, &ops);
      fprintf (stderr, "# ropt GPU: %lu sieve arrays, %lu lines, %.2f s\n",
               calls, ops, sec);
    }
    rsg_ctx_free (ol.ctx);
  }
};
}

/* arrays smaller than this stay on the CPU (the GPU's fixed cost per
   call), larger ones too (memory) */
static size_t ropt_gpu_min_len ()
{
  static const size_t v = [] {
    const char * e = getenv ("CADO_ROPT_GPU_MIN");
    return e ? strtoul (e, nullptr, 10) : (size_t) 1 << 14;
  }();
  return v;
}

void
ropt_sieve_oplist_add (ropt_sieve_oplist * ol, long j, long bound,
                       unsigned int pe, int16_t sub)
{
  if (ol->n == ol->cap) {
    ol->cap = ol->cap ? 2 * ol->cap : 4096;
    ol->buf = rsg_ops_buffer (ol->ctx, ol->cap);
  }
  struct rsg_op & o = ol->buf[ol->n++];
  o.start = j;
  o.bound = bound;
  o.stride = pe;
  o.sub = sub;
}

/* this thread's GPU context, NULL if there is none or it is off */
static ropt_gpu_holder * ropt_gpu_holder_get ()
{
  static const bool enabled = [] {
    const char * e = getenv ("CADO_ROPT_GPU");
    return !(e && e[0] == '0');
  }();
  static thread_local ropt_gpu_holder h;
  if (!enabled)
    return nullptr;
  if (!h.tried) {
    h.tried = true;
    h.ol.ctx = rsg_ctx_create ();
  }
  return h.ol.ctx ? &h : nullptr;
}

ropt_sieve_oplist *
ropt_gpu_begin (size_t len)
{
  if (len < ropt_gpu_min_len () || len > ((size_t) 1 << 26))
    return nullptr;
  ropt_gpu_holder * h = ropt_gpu_holder_get ();
  if (!h)
    return nullptr;
  h->ol.n = 0;
  return &h->ol;
}

static const unsigned short *
ropt_gpu_alpha_lookup (mpz_poly_srcptr f, unsigned long B, void * arg)
{
  auto * a = static_cast<ropt_alpha_batch *> (arg);
  if (B != a->B)
    return nullptr;
  for (size_t i = 0; i < a->fs.size (); i++)
    if (mpz_poly_cmp (f, a->fs[i]) == 0)
      return a->counts.data () + i * a->primes.size ();
  return nullptr;
}

bool
ropt_gpu_alpha_begin (std::vector<cxx_mpz_poly> const & fs)
{
  static const bool enabled = [] {
    const char * e = getenv ("CADO_ROPT_GPU_ALPHA");
    return !(e && e[0] == '0');
  }();
  const unsigned long B = get_alpha_bound ();
  if (!enabled || fs.empty () || B >= 65536)
    return false;
  int deg = 0;
  for (auto const & f : fs) {
    if (f->deg < 1 || f->deg > 31)
      return false;
    deg = std::max (deg, f->deg);
  }
  ropt_gpu_holder * h = ropt_gpu_holder_get ();
  if (!h)
    return false;
  ropt_alpha_batch & a = h->alpha;
  if (a.B != B) {
    /* the primes of get_alpha, in its order */
    a.B = B;
    a.primes.clear ();
    a.primes.push_back (2);
    for (unsigned long p = 3; p <= B; p += 2)
      if (ulong_isprime (p))
        a.primes.push_back (p);
  }
  a.fs = fs;

  size_t nlimbs = 1;
  for (auto const & f : fs)
    for (int i = 0; i <= f->deg; i++)
      nlimbs = std::max (nlimbs, (mpz_sizeinbase (mpz_poly_coeff_const (f, i), 2) + 31) / 32);
  const size_t ncoef = fs.size () * (deg + 1);
  std::vector<uint32_t> limbs (ncoef * nlimbs, 0);
  std::vector<uint8_t> sign (ncoef, 0);
  for (size_t j = 0; j < fs.size (); j++)
    for (int i = 0; i <= fs[j]->deg; i++) {
      mpz_srcptr c = mpz_poly_coeff_const (fs[j], i);
      const size_t at = j * (deg + 1) + i;
      size_t count = 0;
      mpz_export (&limbs[at * nlimbs], &count, -1, sizeof (uint32_t), 0, 0, c);
      sign[at] = mpz_sgn (c) < 0;
    }
  a.counts.resize (fs.size () * a.primes.size ());
  rsg_rootcounts (h->ol.ctx, (int) fs.size (), deg, (int) nlimbs,
                  limbs.data (), sign.data (), (int) a.primes.size (),
                  a.primes.data (), a.counts.data ());

  static const bool check = getenv ("CADO_ROPT_GPU_CHECK") != nullptr;
  if (check) {
    /* the CPU's count, where f mod p is not 0 (get_alpha uses the
       counts only there) */
    gmp_randstate_t rstate;
    gmp_randinit_default (rstate);
    for (size_t j = 0; j < fs.size (); j++)
      for (size_t k = 0; k < a.primes.size (); k++) {
        const unsigned long p = a.primes[k];
        bool zero = true;
        for (int i = 0; i <= fs[j]->deg; i++)
          zero = zero && mpz_divisible_ui_p (mpz_poly_coeff_const (fs[j], i), p);
        if (zero)
          continue;
        const unsigned int n = mpz_poly_roots_ulong (nullptr, fs[j], p, rstate);
        if (n != a.counts[j * a.primes.size () + k]) {
          fprintf (stderr, "# ropt GPU check: %u roots mod %lu, GPU says %u\n",
                   n, p, a.counts[j * a.primes.size () + k]);
          abort ();
        }
      }
    gmp_randclear (rstate);
  }

  set_alpha_rootcount_source (ropt_gpu_alpha_lookup, &a);
  return true;
}

void
ropt_gpu_alpha_end ()
{
  set_alpha_rootcount_source (nullptr, nullptr);
}

void
ropt_gpu_finish (ropt_sieve_oplist * ol, int16_t init, int16_t * array,
                 size_t len)
{
  rsg_apply (ol->ctx, ol->n, init, array, len);

  static const bool check = getenv ("CADO_ROPT_GPU_CHECK") != nullptr;
  if (check) {
    /* the CPU's own loop (rootsieve_run_line) on the same lines */
    std::vector<int16_t> a (len, init);
    for (size_t k = 0; k < ol->n; k++) {
      const struct rsg_op & o = ol->buf[k];
      for (long j = o.start; j <= o.bound; j += o.stride)
        a[j] = a[j] - (int16_t) o.sub;
    }
    size_t bad = 0;
    for (size_t j = 0; j < len; j++)
      bad += a[j] != array[j];
    if (bad) {
      fprintf (stderr, "# ropt GPU check: %zu of %zu entries differ\n",
               bad, len);
      abort ();
    }
  }
}

#else

void
ropt_sieve_oplist_add (ropt_sieve_oplist *, long, long, unsigned int,
                       int16_t)
{
  ASSERT_ALWAYS (0);
}

ropt_sieve_oplist * ropt_gpu_begin (size_t) { return nullptr; }

bool ropt_gpu_alpha_begin (std::vector<cxx_mpz_poly> const &) { return false; }

void ropt_gpu_alpha_end () {}

void
ropt_gpu_finish (ropt_sieve_oplist *, int16_t, int16_t *, size_t)
{
  ASSERT_ALWAYS (0);
}

#endif
