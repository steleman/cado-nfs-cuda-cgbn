#ifndef CADO_COFACTORING_METHOD_HPP
#define CADO_COFACTORING_METHOD_HPP

#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/ostream.h"

#include "facul_ecm.h"
#include "facul_method.hpp"

/* A factoring method, together with what a benchmark measured about it.
 *
 * Only the method family and the bounds B1 and B2 are part of the
 * measured data. The ECM parameter (sigma) is not: it is drawn when the
 * method is instantiated for a bench or for a real factorization, so
 * params.parameter is left at 0 here and carries no meaning.
 */
/* The name the files in this directory give a method: the one facul
 * uses, which already says which ECM parameterization it is, so that
 * nothing has to carry a separate curve column.
 *
 * The generator also has null methods, placeholders for "nothing in this
 * slot", which facul has no name for. They are written as "none". */
std::string cofactoring_method_name(facul_method::parameters const & p);

/* The inverse. Returns false on anything that is not a method name. */
bool cofactoring_method_by_name(std::string const & name,
                                facul_method::parameters & p);

struct cofactoring_method {
    facul_method::parameters params;

    /* proba[i] is the probability that this method finds a prime factor
     * of len_p_min + i bits. */
    unsigned int len_p_min = 0;
    std::vector<double> proba;

    /* time[i] is the time this method takes on an input whose size puts
     * it in bucket i; see time_index() below. */
    std::vector<double> time;

    cofactoring_method() = default;
    explicit cofactoring_method(facul_method::parameters p)
        : params(std::move(p))
    {
    }

    /* Build the parameters from the method, its curve and its bounds,
     * which is how the generator passes them around. */
    static facul_method::parameters
    make_parameters(facul_method_code method, ec_parameterization_t curve,
                    unsigned long B1, unsigned long B2);

    /* from the four numbers of one record of the file format */
    static cofactoring_method from_fields(unsigned long method,
                                        unsigned long curve, unsigned long B1,
                                        unsigned long B2)
    {
        return cofactoring_method(make_parameters(facul_method_code(method),
                                                ec_parameterization_t(curve),
                                                B1, B2));
    }

    facul_method_code method() const { return params.method; }
    unsigned long B1() const { return params.B1; }
    unsigned long B2() const { return params.B2; }

    /* 0 for everything but ECM, matching what the file format stores. */
    ec_parameterization_t curve() const
    {
        return params.method == EC_METHOD ? params.parameterization
                                          : ec_parameterization_t(0);
    }

    /* The generator uses methods with B1 == B2 == 0 as placeholders for
     * "no method in this slot". */
    bool is_zero() const { return params.is_null(); }
    void put_zero();

    /* Only the method and its bounds are compared, not the measurements:
     * this answers "are these the same method?". */
    bool same_method_as(cofactoring_method const & o) const;

    /* time[] is indexed by the number of half-words of the input, the
     * way bench_time() measures it: bucket 0 covers anything up to one
     * word, then one bucket per extra half-word. Returns an index that
     * may be past the end of a short time[]; callers clamp. */
    static unsigned int time_index(unsigned int r);

    /* time[] entry to use for an r-bit input, clamped to what we have. */
    double time_for(unsigned int r) const;
};

std::istream & operator>>(std::istream & is, cofactoring_method &);
std::ostream & operator<<(std::ostream & os, cofactoring_method const &);

namespace fmt
{
template <> struct formatter<cofactoring_method> : ostream_formatter {
};
} // namespace fmt

using cofactoring_method_list = std::vector<cofactoring_method>;

/* The methods of one family. curve is ignored unless method is
 * EC_METHOD. */
cofactoring_method_list extract_fm_method(cofactoring_method_list const & t, facul_method_code method,
                             ec_parameterization_t curve);

/* Positive if el1 sorts after el2. Beware: this is not a strict weak
 * ordering -- it never reports two methods as equivalent, and it sums
 * signed differences rather than comparing them. It is exposed only so
 * that a test can pin the order that sort_by_proba produces. */
int fm_compare_by_proba(cofactoring_method const & el1,
                        cofactoring_method const & el2);

/* Sort by increasing probability, zero methods first. */
void sort_by_proba(cofactoring_method_list & t);

std::istream & operator>>(std::istream & is, cofactoring_method_list &);
std::ostream & operator<<(std::ostream & os, cofactoring_method_list const &);

namespace fmt
{
template <> struct formatter<cofactoring_method_list> : ostream_formatter {
};
} // namespace fmt


#endif /* CADO_COFACTORING_METHOD_HPP */
