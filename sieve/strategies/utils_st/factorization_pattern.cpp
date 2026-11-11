#include "cado.h" // IWYU pragma: keep

#include <algorithm>
#include <istream>
#include <limits>
#include <ostream>
#include <utility>

#include "factorization_pattern.hpp"
#include "utils_cxx.hpp"

std::ostream & operator<<(std::ostream & o, factorization_pattern const & P)
{
    return o << "[ " << join(P, " ") << " ]: " << P.count;
}

std::istream & operator>>(std::istream & is, factorization_pattern & P)
{
    P.clear();
    is >> expect("[");
    for (unsigned int v; is.good() && is >> std::ws && is.peek() != ']';) {
        is >> v;
        P.push_back(v);
    }
    return is >> expect("]:") >> P.count;
}

bool factorization_pattern::operator<(factorization_pattern const & o) const
{
    if (!std::ranges::is_sorted(*this)) {
        factorization_pattern a = *this;
        std::ranges::sort(a);
        return a < o;
    }
    if (!std::ranges::is_sorted(o)) {
        factorization_pattern b = o;
        std::ranges::sort(b);
        return *this < b;
    }
    bool c = std::ranges::lexicographical_compare(*this, o);
    if (size() != o.size())
        return c;
    if (!std::ranges::equal(*this, o))
        return c;
    return count < o.count - 0.1;
}

std::ostream & operator<<(std::ostream & os,
                          factorization_pattern_list const & t)
{
    for (auto const & P: t)
        os << P << "\n";
    return os;
}

std::istream & operator>>(std::istream & is, factorization_pattern_list & t)
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
        factorization_pattern P;
        /* Without this check a malformed record loops here forever: the
         * failed stream never reaches eof, and every turn appends another
         * empty pattern. */
        if (!(is >> P))
            return is;
        t.push_back(std::move(P));
    }
    return is;
}
