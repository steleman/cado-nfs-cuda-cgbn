#include "cado.h" // IWYU pragma: keep

#include <cstdlib>

#include <istream>
#include <ostream>
#include <string>
#include <memory>
#include <ios>
#include <iterator>
#include <limits>
#include <locale>
#include <streambuf>

#include <gmp.h>
#include "gmpxx.hpp"

#include "utils_cxx.hpp"

using namespace std;

/* In case we don't have gmpxx, we have to provide the C++ I/O functions
 * by ourselves...
 *
 * We make no effort to do this very accurately, though.
 */

namespace {
inline int getbase(ostream const& o)
{
    switch(o.flags() & std::ios_base::basefield) {
        case std::ios::hex:
            return 16;
        case std::ios::dec:
            return 10;
        case std::ios::oct:
            return 8;
        default:
            return 10;
    }
}
} // namespace

ostream& operator<<(ostream& os, mpz_srcptr x)
{
    int b = getbase(os);
    const unique_ptr<char[], free_delete<char>> str(mpz_get_str(nullptr, b, x));
    if ((os.flags() & std::ios::showbase) && (b == 8 || b == 16)) {
        if (str[0] == '-')
            os << "-0" << (b == 8 ? 'b' : 'x') << str.get() + 1;
        else
            os << "0" << (b == 8 ? 'b' : 'x') << str.get();
    } else {
        os << str.get();
    }
    return os;
}

ostream& operator<<(ostream& os, mpq_srcptr x)
{
    const unique_ptr<char, free_delete<char>> str(mpq_get_str(nullptr, getbase(os), x));
    os << str.get();
    return os;
}

namespace {
/* a stream buffer that reads from another one, without buffering, and
 * keeps what it consumes */
struct recording_streambuf : public std::streambuf {
    std::streambuf * src;
    string text;
    explicit recording_streambuf(std::streambuf * src) : src(src) {}

  protected:
    int_type underflow() override { return src->sgetc(); }
    int_type uflow() override
    {
        int_type const c = src->sbumpc();
        if (!traits_type::eq_int_type(c, traits_type::eof()))
            text.push_back(traits_type::to_char_type(c));
        return c;
    }
};
} // namespace

/* We read exactly what is >> (long &) would read: the num_get facet of
 * the stream decides which characters make the integer and sets the
 * stream state.
 * Only when the value does not fit in a long do we convert the
 * characters that num_get consumed ourselves.
 */
istream& operator>>(istream& is, mpz_ptr x)
{
    istream::sentry const s(is);
    if (!s)
        return is;
    recording_streambuf rb(is.rdbuf());
    ios_base::iostate err = ios_base::goodbit;
    long v;
    std::use_facet<std::num_get<char>>(is.getloc()).get(
            std::istreambuf_iterator<char>(&rb), {}, is, err, v);
    string const & text = rb.text;
    if ((err & ios_base::failbit)
            && (v == std::numeric_limits<long>::max()
                || v == std::numeric_limits<long>::min())) {
        /* overflow: text is a complete integer */
        char const * t = text.c_str();
        bool const neg = *t == '-';
        if (neg || *t == '+')
            t++;
        int base = 0;
        switch (is.flags() & ios_base::basefield) {
            case ios_base::hex:
                base = 16;
                if (t[0] == '0' && (t[1] == 'x' || t[1] == 'X'))
                    t += 2;
                break;
            case ios_base::oct:
                base = 8;
                break;
            case ios_base::dec:
                base = 10;
                break;
            default:
                break;
        }
        if (mpz_set_str(x, t, base) == 0) {
            err &= ~ios_base::failbit;
            if (neg)
                mpz_neg(x, x);
        }
    } else {
        mpz_set_si(x, v);
    }
    is.setstate(err);
    return is;
}


istream& operator>>(istream& is, mpq_ptr x)
{
    string s;
    is >> s;
    mpq_set_str(x, s.c_str(), 0);
    return is;
}

