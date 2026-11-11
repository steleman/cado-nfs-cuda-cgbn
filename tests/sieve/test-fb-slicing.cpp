#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include "fmt/base.h"

#include "cado_poly.hpp"
#include "fb-types.hpp"
#include "fb.hpp"
#include "las-config.hpp"
#include "las-side-config.hpp"
#include "macros.h"
#include "params.hpp"
#include "verbose.hpp"

/* Check that all entries of a slice are sieved with the logp of that
 * slice. This is the case by construction for the entries with a fixed
 * number of roots, which are primes sorted by increasing p. The general
 * entries, in contrast, are sorted by increasing q = p^k but they are
 * sieved with the logp of p^exp / p^oldexp, which is not monotonic in q.
 */

static unsigned char entry_logp(fb_entry_general const & e, double scale)
{
#ifdef BUCKET_SIEVE_POWERS
    return fb_log_delta(e.p, e.roots[0].exp, e.roots[0].oldexp, scale);
#else
    return fb_log(e.get_q(), scale, 0);
#endif
}

static int check(fb_factorbase & fb, fbprime_t lim, fbprime_t bkthresh,
                 fbprime_t bkthresh1, double scale)
{
    fb_factorbase::key_type K;
    K.thresholds = {{ bkthresh, bkthresh1, lim, lim }};
    K.td_thresh = 1024;
    K.scale = scale;
    K.nb_threads = 1;

    fb_factorbase::slicing const & S = fb[K];

    size_t nentries = 0, nslices = 0, nbad = 0;
    for (size_t i = 1; i < S.nparts(); i++) {
        for (auto const & s: S.get_part(i).slices.get<-1>()) {
            nslices++;
            for (auto const & e: s) {
                nentries++;
                nbad += entry_logp(e, scale) != s.get_logp();
            }
        }
    }
    fmt::print("bkthresh={} bkthresh1={} scale={}: {} general entries"
               " in {} slices, {} with a wrong logp\n",
               bkthresh, bkthresh1, scale, nentries, nslices, nbad);
    return nentries == 0 || nbad != 0;
}

int main(int argc, char const * argv[])
{
    cxx_param_list pl;
    pl.declare_usage("poly", "polynomial file");
    siever_side_config::declare_usage(pl);
    verbose_decl_usage(pl);
    pl.process_command_line(argc, argv, false);
    verbose_interpret_parameters(pl);

    cxx_cado_poly const cpoly(pl);

    /* side 0 must be a rational side, whose factor base is computed
     * here, with all prime powers below lim */
    ASSERT_ALWAYS(cpoly[0]->deg == 1);
    unsigned long lim = 0;
    if (!pl.parse("lim0", lim))
        pl.fail("missing -lim0");

    fb_factorbase fb(cpoly, 0, pl, nullptr, 1);

    int failures = 0;
    for (double const scale: { 1.0, 1.4, 1.62, 2.0 }) {
        failures += check(fb, lim, 1 << 12, 1 << 17, scale);
        failures += check(fb, lim, 1 << 13, 1 << 16, scale);
    }

    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
