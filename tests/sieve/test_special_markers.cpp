#include "cado.h" // IWYU pragma: keep

/* The lines that las prints and other programs read back
 * (las-special-markers.hpp) must parse back to what was printed. */

#include <cstdlib>

#include <string>

#include "fmt/format.h"

#include "cxx_mpz.hpp"
#include "las-qlattice.hpp"
#include "las-special-markers.hpp"
#include "las-sublat.hpp"
#include "macros.h"
#include "special-q.hpp"
#include "utils_cxx.hpp"

namespace markers = cado::special_markers;

template <typename T> static T roundtrip(T const & m)
{
    std::string const line = fmt::format("{}", m);
    auto back = T::parse(line);
    if (!back)
        throw cado::error("cannot parse back: {}", line);
    return *back;
}

int main()
{
    special_q const prime(cxx_mpz(1000003), cxx_mpz(4242), 1);
    special_q const composite(cxx_mpz(1009 * 1013), cxx_mpz(17), 0);

    for (auto const & Q: { prime, composite }) {
        ASSERT_ALWAYS(roundtrip(markers::begin_special_q { Q }).Q == Q);
        ASSERT_ALWAYS(roundtrip(markers::end_special_q { Q }).Q == Q);

        auto const b = roundtrip(markers::batch_relations { 12, Q });
        ASSERT_ALWAYS(b.n == 12 && b.Q == Q);

        qlattice_basis B(-7, 3, 11, -2);
        B.doing = Q;
        auto const s = roundtrip(markers::sieving<qlattice_basis> {
                B, 1024, 512, " # extra" });
        ASSERT_ALWAYS(s.Q.doing == Q);
        ASSERT_ALWAYS(s.Q.a0 == -7 && s.Q.b0 == 3 && s.Q.a1 == 11
                && s.Q.b1 == -2);
        ASSERT_ALWAYS(s.I == 1024 && s.J == 512 && s.extra == " # extra");
    }

    {
        siqs_special_q_data D;
        D.doing = composite;
        D.crt_data_modq = { cxx_mpz(123), cxx_mpz(456) };
        auto const s = roundtrip(markers::sieving<siqs_special_q_data> {
                D, 64, 32 });
        ASSERT_ALWAYS(s.Q.doing == composite);
        ASSERT_ALWAYS(s.Q.crt_data_modq == D.crt_data_modq);
        ASSERT_ALWAYS(s.I == 64 && s.J == 32 && s.extra.empty());
    }

    auto const sl = roundtrip(markers::begin_sublattice {
            sublat_runtime_t(3, 2, 1) }).sl;
    ASSERT_ALWAYS(sl.m == 3 && sl.i0 == 2 && sl.j0 == 1);

    /* what las prints, as the scripts see it */
    ASSERT_ALWAYS(fmt::format("{}", markers::begin_special_q { prime })
            == "# Now sieving side-1 q=1000003; rho=4242");
    ASSERT_ALWAYS(fmt::format("{}", markers::end_special_q { composite })
            == "# Time for side-0 q=1022117=1009*1013; rho=17:");
    ASSERT_ALWAYS(fmt::format("{}", markers::batch_relations { 3, prime })
            == "# 3 relations for side-1 q=1000003; rho=4242");
    {
        qlattice_basis B(-7, 3, 11, -2);
        B.doing = prime;
        ASSERT_ALWAYS(fmt::format("{}", markers::sieving<qlattice_basis> {
                        B, 1024, 512 })
                == "# Sieving side-1 q=1000003; rho=4242;"
                   " a0=-7; b0=3; a1=11; b1=-2; I=1024; J=512;");
    }
    ASSERT_ALWAYS(fmt::format("{}", markers::begin_sublattice {
                    sublat_runtime_t(2, 0, 1) })
            == "# Sublattice (i, j) == (0, 1) mod 2");

    /* the end of a line, as las prints it after the marker */
    {
        auto const e = markers::end_special_q::parse(
                "# Time for side-1 q=1000003; rho=4242: (1.2 elapsed) 3.40s");
        ASSERT_ALWAYS(e && e->Q == prime);
    }
    /* older las output */
    {
        auto const e = markers::end_special_q::parse(
                "# Time for side-1 q=62627 r=34758: 0.01s [norm 0.00+0.0]");
        ASSERT_ALWAYS(e && e->Q.p == 62627 && e->Q.r == 34758);
    }
    /* other lines */
    ASSERT_ALWAYS(!markers::end_special_q::parse(
                "# Time for 1000 random lookups: 1.2s"));
    ASSERT_ALWAYS(!markers::begin_special_q::parse("# Now sieving"));
    ASSERT_ALWAYS(!markers::begin_special_q::parse("# Sieving side-1 q=3; rho=1"));
    ASSERT_ALWAYS(!markers::sieving<qlattice_basis>::parse(
                "# Now sieving side-1 q=3; rho=1"));
    /* a basis that is cut short */
    ASSERT_ALWAYS(!markers::sieving<qlattice_basis>::parse(
                "# Sieving side-1 q=1000003; rho=4242; a0=-7; b0=3; I=1024;"));
    ASSERT_ALWAYS(!markers::batch_relations::parse(
                "# 12 relation for side-1 q=3; rho=1"));
    ASSERT_ALWAYS(!markers::batch_relations::parse(
                "# x relations for side-1 q=3; rho=1"));
    ASSERT_ALWAYS(!markers::begin_sublattice::parse(
                "# Sublattice (i, j) == (0, 1) mod 2 trailing"));
    ASSERT_ALWAYS(!markers::begin_special_q::parse("-1,2:3:4"));

    return EXIT_SUCCESS;
}
