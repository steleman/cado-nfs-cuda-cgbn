#include "cado.h" // IWYU pragma: keep

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

#include <istream>
#include <ostream>
#include <utility>

#include "arith/modredc_ul.h" // MODREDCUL_MAXBITS

#include "facul_strategies.hpp"
#include <string>
#include "cofactoring_method.hpp"
#include "utils_cxx.hpp"

facul_method::parameters
cofactoring_method::make_parameters(facul_method_code method,
                                  ec_parameterization_t curve,
                                  unsigned long B1, unsigned long B2)
{
    /* Only ECM has a curve. The files used to carry one next to every
     * method, filled by the survey with whichever parameterization its
     * sweep happened to be on, and meaningless for PM1 and PP1; they
     * name the method instead now, so there is nothing to carry and the
     * slot keeps its default. */
    if (method == EC_METHOD)
        return {method, B1, B2, curve, 0, 1};
    return {method, B1, B2};
}

void cofactoring_method::put_zero()
{
    params.B1 = 0;
    params.B2 = 0;
    for (auto & x: proba)
        x = 0;
    for (auto & x: time)
        x = 0;
}

bool cofactoring_method::same_method_as(cofactoring_method const & o) const
{
    return params.method == o.params.method &&
           params.parameterization == o.params.parameterization &&
           params.B1 == o.params.B1 && params.B2 == o.params.B2;
}

unsigned int cofactoring_method::time_index(unsigned int r)
{
    /* We add 0.5 to the length of one word, because our times are
     * measured for an inclusive length: with MODREDCUL_MAXBITS = 64, a
     * cofactor fits in one word when its length is less than *or equal
     * to* 64 bits. Without the 0.5 the equality case would land in the
     * next bucket. */
    double const half_word = (MODREDCUL_MAXBITS + 0.5) / 2.0;
    unsigned int const number_half_wd = (unsigned int) floor(r / half_word);
    return number_half_wd < 2 ? 0 : number_half_wd - 1;
}

double cofactoring_method::time_for(unsigned int r) const
{
    unsigned int const i = time_index(r);
    return i >= time.size() ? time.back() : time[i];
}

/* The placeholder the generator uses for an empty slot. facul has no
 * name for it, since it is not one of its methods. */
static const char * null_method_name = "none";

std::string cofactoring_method_name(facul_method::parameters const & p)
{
    if (p.method == NO_METHOD)
        return null_method_name;
    return facul_method_name(p);
}

bool cofactoring_method_by_name(std::string const & name,
                                facul_method::parameters & p)
{
    if (name == null_method_name) {
        p = facul_method::parameters();
        return true;
    }
    return facul_method_by_name(name, p);
}

std::ostream & operator<<(std::ostream & os, cofactoring_method const & fm)
{
    /* The method's name and its two bounds, then the probabilities
     * behind the bit size they start at, then the timings, each field
     * closed by a '|'. The name says which ECM parameterization this is,
     * so there is no separate curve column. */
    os << cofactoring_method_name(fm.params) << ' '
       << fm.params.B1 << ' ' << fm.params.B2 << " | ";
    os << fm.len_p_min << ' ';
    for (double const p: fm.proba)
        os << fmt::format("{:f} ", p);
    os << "| ";
    for (double const t: fm.time)
        os << fmt::format("{:f} ", t);
    os << "|\n";
    return os;
}

/* read doubles until the next '|' */
static bool read_doubles(std::istream & is, std::vector<double> & v)
{
    for (;;) {
        is >> std::ws;
        if (!is.good() || is.peek() == '|')
            break;
        double x;
        if (!(is >> x))
            return false;
        v.push_back(x);
    }
    return true;
}

std::istream & operator>>(std::istream & is, cofactoring_method & fm)
{
    std::string name;
    unsigned long B1, B2;
    if (!(is >> name >> B1 >> B2))
        return is;

    facul_method::parameters params;
    if (!cofactoring_method_by_name(name, params)) {
        is.setstate(std::ios_base::failbit);
        return is;
    }

    if (!(is >> std::ws >> expect("|")))
        return is;

    unsigned int len_p_min;
    if (!(is >> len_p_min))
        return is;

    std::vector<double> proba;
    std::vector<double> time;

    if (!read_doubles(is, proba) || !(is >> std::ws >> expect("|")))
        return is;
    if (!read_doubles(is, time) || !(is >> std::ws >> expect("|")))
        return is;

    fm.params = cofactoring_method::make_parameters(
        params.method, params.parameterization, B1, B2);
    fm.len_p_min = len_p_min;
    fm.proba = std::move(proba);
    fm.time = std::move(time);
    return is;
}

static double const EPSILON_DBL = 0.000001;

cofactoring_method_list extract_fm_method(cofactoring_method_list const & t, facul_method_code method,
                             ec_parameterization_t curve)
{
    cofactoring_method_list res;
    for (auto const & fm: t) {
        if (fm.params.method != method)
            continue;
        if (method == EC_METHOD && fm.params.parameterization != curve)
            continue;
        res.push_back(fm);
    }
    return res;
}

/************************************************************************/
/*                      SORT                                            */
/************************************************************************/

/* Positive if el1 is "greater" than el2.
 *
 * Note that this is not a strict weak ordering: it never reports two
 * methods as equivalent, and it sums signed differences rather than
 * comparing them. Replacing it, or handing it to std::sort, changes
 * which order comes out, and with it every file this directory produces.
 * The selection sort below is part of the same bargain. */
int fm_compare_by_proba(cofactoring_method const & el1,
                        cofactoring_method const & el2)
{
    if (el1.is_zero())
        return -1;
    if (el2.is_zero())
        return 1;
    /* assumes that len_p_min is the same for both */
    size_t const len = std::min(el1.proba.size(), el2.proba.size());
    double diff_proba = 0;
    for (size_t i = 0; i < len; i++)
        if (el1.proba[i] > EPSILON_DBL && el2.proba[i] > EPSILON_DBL)
            diff_proba += el1.proba[i] - el2.proba[i];

    return (diff_proba > EPSILON_DBL) ? 1 : -1;
}

void sort_by_proba(cofactoring_method_list & t)
{
    for (size_t max = t.size(); max > 0; max--) {
        size_t index_max = 0;
        for (size_t i = 0; i < max; i++)
            if (fm_compare_by_proba(t[i], t[index_max]) > 0)
                index_max = i;
        std::swap(t[max - 1], t[index_max]);
    }
}

/************************************************************************/
/*                      I/O                                             */
/************************************************************************/

std::ostream & operator<<(std::ostream & os, cofactoring_method_list const & t)
{
    for (auto const & fm: t)
        os << fm;
    return os;
}

std::istream & operator>>(std::istream & is, cofactoring_method_list & t)
{
    for (;;) {
        is >> std::ws;
        if (is.eof())
            break;
        if (!is.good())
            return is;
        if (is.peek() == '#') {
            is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        cofactoring_method fm;
        if (!(is >> fm))
            return is;
        t.push_back(std::move(fm));
    }
    return is;
}
