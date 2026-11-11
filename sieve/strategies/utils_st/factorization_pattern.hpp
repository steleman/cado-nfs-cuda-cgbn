#ifndef FACTORIZATION_PATTERN_HPP
#define FACTORIZATION_PATTERN_HPP

#include <istream>
#include <ostream>
#include <vector>

#include "fmt/base.h"
#include "fmt/ostream.h"
#include "cofactoring_strategy.hpp"

/* One way a cofactor can factor, described only by the sizes involved:
 * the bit size of each of its prime factors, and an estimate of how many
 * integers of the size we are interested in factor that way.
 *
 * For instance [ 20 21 ]: 1.12e9 says that about 1.12e9 of the integers
 * under consideration are the product of a 20-bit and a 21-bit prime.
 *
 * The target size itself is not stored here: it is a property of the
 * whole list, not of one pattern, and it cannot be read off the summands
 * (a 20-bit times a 21-bit prime has 40 or 41 bits).
 */
struct factorization_pattern : public std::vector<unsigned int> {
    /* how many integers are expected to match this pattern */
    double count = 0;

    template <typename... Args>
    explicit factorization_pattern(double n, Args &&... args)
        : std::vector<unsigned int>(std::forward<Args>(args)...)
        , count(n)
    {
    }
    explicit factorization_pattern(double n, std::vector<unsigned int> v)
        : std::vector<unsigned int>(std::move(v))
        , count(n)
    {
    }
    factorization_pattern() = default;
    ~factorization_pattern() = default;
    factorization_pattern(factorization_pattern const &) = default;
    factorization_pattern(factorization_pattern &&) = default;
    factorization_pattern & operator=(factorization_pattern const &) = default;
    factorization_pattern & operator=(factorization_pattern &&) = default;

    bool operator<(factorization_pattern const &) const;
    bool operator==(factorization_pattern const & o) const
    {
        return !operator!=(o);
    }
    bool operator!=(factorization_pattern const & o) const
    {
        return *this < o || o < *this;
    }
};

/* A list of patterns, all for cofactors of the same size. */
using factorization_pattern_list = std::vector<factorization_pattern>;

/* Are all the prime factors of this pattern within the size range that a
 * strategy can hope to find? */
inline bool all_sizes_within(factorization_pattern const & P,
                             unsigned int min_bits, unsigned int max_bits)
{
    for (auto x: P)
        if (x > max_bits || x < min_bits)
            return false;
    return true;
}

std::istream & operator>>(std::istream & is, factorization_pattern &);
std::ostream & operator<<(std::ostream & o, factorization_pattern const &);

std::istream & operator>>(std::istream & is, factorization_pattern_list &);
std::ostream & operator<<(std::ostream & o, factorization_pattern_list const &);

namespace fmt
{
template <> struct formatter<factorization_pattern> : ostream_formatter {
};
template <> struct formatter<factorization_pattern_list> : ostream_formatter {
};
} // namespace fmt

#endif /* FACTORIZATION_PATTERN_HPP */
