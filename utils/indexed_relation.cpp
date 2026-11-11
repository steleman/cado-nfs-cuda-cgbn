#include "cado.h" // IWYU pragma: keep

#include <cstddef>

#include <algorithm>
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <ios>

#include <gmp.h>
#include "fmt/base.h"
#include "fmt/format.h"

#include "gmp_aux.h"
#include "indexed_relation.hpp"
#include "relation.hpp"
#include "relation-tools.h"
#include "renumber.hpp"
#include "typedefs.h"

template<typename Storage>
indexed_relation_tmpl<Storage>::indexed_relation_tmpl(relation const & rel, renumber_t const & R)
    : relation_ab((relation_ab const &) rel)
{
    Storage::set_active_sides(rel.active_sides);
    for(unsigned int side_index = 0 ; side_index < 2 ; side_index++) {
        int const side = rel.active_sides[side_index];
        for(auto const & pr : rel.sides[side]) {
            p_r_values_t const p = mpz_get_ui(pr.p);
            p_r_values_t r = mpz_get_ui(pr.r);
            if ((int) side == rel.rational_side)
                r = relation_compute_r(a, b, p);
            renumber_t::p_r_side const ipr { p, r, side };

            if (R.is_bad(ipr)) {
                auto ie = R.indices_from_p_a_b(ipr, pr.e, a, b);
                for(size_t i = 0 ; i < ie.second.size() ; i++)
                    for(int k = ie.second[i] ; k-- ; )
                        (*this)[side_index].push_back(ie.first + i);
            } else {
                index_t const i = R.index_from_p_r(ipr);
                for(int k = pr.e ; k-- ; )
                    (*this)[side_index].push_back(i);
            }
        }
    }
}

template<typename Storage>
void indexed_relation_tmpl<Storage>::sort() {
    for(auto & s : Storage::containers())
        std::ranges::sort(s);
}

template<typename Storage>
void 
indexed_relation_tmpl<Storage>::shrink(double shrink_factor, index_t shrink_threshold) {
    // Indices below shrink_threshold are not shrunk
    // FIXME: I am not sure we should keep the heavy weight columns
    // un-shrunk. The answer might be different in DL and in facto...
    for(auto & s : Storage::containers()) {
        for(auto & i : s) {
            if (i >= shrink_threshold) {
                i = shrink_threshold + (index_t)
                    ((i - shrink_threshold) / shrink_factor);
            }
        }
    }
}

template<typename Storage>
void indexed_relation_tmpl<Storage>::compress(bool dl)
requires (!Storage::separates_sides)
{
    /* sort() must be called first ! */
    if (dl) return;
    auto & data = *Storage::containers().begin();
    auto w = data.begin();
    for(auto r = data.begin(), s = r ; r != data.end() ; r = s) {
        for(++s ; s != data.end() && *r == *s ; ++s);
        if ((s - r) & 1)
            *w++ = *r;
    }
    data.erase(w, data.end());
}
template<typename Storage>
void indexed_relation_tmpl<Storage>::compress(bool dl)
requires Storage::separates_sides
{
    auto & sides = Storage::containers();
    static_assert(std::tuple_size_v<std::decay_t<decltype(sides)>> == 2);

    /* sort() must be called first ! */
    if (dl) {
        /* Exponents are written in unary here, so a repeated index is
         * legitimate. But filter/README.fileformat says that all
         * appearances of an index must be contiguous, and the two sides
         * are printed one after the other. Hence the merge. */
        auto const n0 = std::ptrdiff_t(sides[0].size());
        sides[0].insert(sides[0].end(), sides[1].begin(), sides[1].end());
        std::inplace_merge(sides[0].begin(), sides[0].begin() + n0,
                sides[0].end());
        sides[1].clear();
        return;
    }

    auto r0 = sides[0].begin(), w0 = r0;
    auto r1 = sides[1].begin(), w1 = r1;
    for( ; r0 != sides[0].end() || r1 != sides[1].end() ; ) {
        if (r1 == sides[1].end() || (r0 != sides[0].end() && *r0 < *r1)) {
            auto s0 = r0;
            for(++s0 ; s0 != sides[0].end() && *s0 == *r0 ; ++s0);
            if ((s0 - r0) & 1)
                *w0++ = *r0;
            r0 = s0;
        } else if (r0 == sides[0].end() || *r1 < *r0) {
            auto s1 = r1;
            for(++s1 ; s1 != sides[1].end() && *s1 == *r1 ; ++s1);
            if ((s1 - r1) & 1)
                *w1++ = *r1;
            r1 = s1;
        } else {
            ASSERT_ALWAYS(*r0 == *r1);
            auto s0 = r0;
            for(++s0 ; s0 != sides[0].end() && *s0 == *r0 ; ++s0);
            auto s1 = r1;
            for(++s1 ; s1 != sides[1].end() && *s1 == *r1 ; ++s1);
            if (((s1 - r1) + (s0 - r0)) & 1)
                *w0++ = *r0;
            r0 = s0;
            r1 = s1;
        }
    }
    sides[0].erase(w0, sides[0].end());
    sides[1].erase(w1, sides[1].end());
}

template<typename Storage>
std::ostream& operator<<(std::ostream& os, indexed_relation_tmpl<Storage> const & rel)
{
#if 0
    os << fmt::format("{:x},{:x}", (uint64_t) rel.a, rel.b);
    /* this sorts the indices before printing. A priori there's no real
     * point in doing that, but just in case.
     *
     * The old fake_rels binary was doing that, though. And it was also
     * printing rel.a as unsigned.
     */
    std::vector<index_t> S;
    char c = ':';
    for(auto const & s : rel.sides) {
        std::ranges::copy(s, std::back_inserter(S));
    }
    std::ranges::sort(S);

        for(auto i : S) {
            os << c << fmt::format("{:x}", i);
            c = ',';
        }
#else
    os << fmt::format("{:x},{:x}", rel.a, rel.b);
    char c = ':';
    for(auto const & s : rel.containers()) {
        for(auto i : s) {
            os << c << fmt::format("{:x}", i);
            c = ',';
        }
    }
#endif
    return os;
}

template<typename Storage>
std::istream& operator>>(std::istream& is, indexed_relation_tmpl<Storage>& rel)
{
    std::string s;
    if (!std::getline(is, s, '\n') || !rel.parse(rel, s.c_str())) {
        is.setstate(std::ios_base::failbit);
        rel = indexed_relation_tmpl<Storage>();
    }
    return is;
}


int 
indexed_relation_normal_storage::parse(relation_ab & ab, const char *line)
{
    int consumed;

    if (gmp_sscanf(line, "%Zx,%Zx%n",
                (mpz_ptr) ab.az, (mpz_ptr) ab.bz,
                &consumed) < 2)
        return 0;

    ab.a = mpz_get_int64(ab.az);
    ab.b = mpz_get_uint64(ab.bz);

    if (line[consumed] == '@') {
        consumed++;
        int c;
        if (sscanf(line + consumed, "%u,%u%n",
                &ab.active_sides[0],
                &ab.active_sides[1],
                &c) < 2)
            return 0;
        consumed += c;
    } else {
        ab.active_sides[0] = 0;
        ab.active_sides[1] = 1;
    }

    if (line[consumed] != ':')
        return 0;
    consumed++;

    while(line[consumed] != '\0' && line[consumed] != '\n') {
        index_t p;
        int consumed_p;
        if (sscanf(line + consumed, "%" SCNid "%n", &p, &consumed_p) < 1)
            return 0;
        (*this)[0].push_back(p);
        consumed += consumed_p;
        if (line[consumed] == ',')
            consumed++;
    }
    return 1;
}

template struct indexed_relation_tmpl<indexed_relation_normal_storage>;
template struct indexed_relation_tmpl<indexed_relation_byside_storage>;

template 
std::ostream& operator<<(std::ostream& os, indexed_relation_tmpl<indexed_relation_byside_storage> const & rel);
template 
std::ostream& operator<<(std::ostream& os, indexed_relation_tmpl<indexed_relation_normal_storage> const & rel);

template
std::istream& operator>>(std::istream& is, indexed_relation_tmpl<indexed_relation_normal_storage>& rel);

