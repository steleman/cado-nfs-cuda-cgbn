#include "cado.h" // IWYU pragma: keep

#include <cstdint>

#include <ios>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gmp.h>
#include "fmt/base.h"
#include "fmt/format.h"

#include "cxx_mpz.hpp"
#include "gmp_auxx.hpp"
#include "macros.h"

/* we've had this cxx_mpz type around for years, and yet no proper
 * unit testing is in place. At the very least, the specifics of fmt
 * printing deserve their own test.
 */

static void check(long z, fmt::format_string<cxx_mpz> const & format, std::string const & fcopy, std::string const & exp)
{
    cxx_mpz zz = z;
    std::string gotz = fmt::format(fmt::runtime(format), z);
    std::string gotzz = fmt::format(fmt::runtime(format), zz);
    if (gotz == exp && gotzz == exp)
        return;
    throw cado::error("format error while printing {} with format {}:"
            " long -> {} ; cxx_mpz -> {} ; expected {}",
            z, fcopy, gotz, gotzz, exp);
}

#define checkcheck(a,b,c) check(a,b,b,c)

/* integral on the left of a binary operator with a cxx_mpz. We compare
 * with what we get after converting the integral to cxx_mpz first.
 */
template<typename T>
static void check_left_operand(T a, long b)
{
    const cxx_mpz za = a;
    const cxx_mpz zb = b;
    ASSERT_ALWAYS(a + zb == za + zb);
    ASSERT_ALWAYS(a - zb == za - zb);
    ASSERT_ALWAYS(a * zb == za * zb);
    if (b) {
        ASSERT_ALWAYS(a / zb == za / zb);
        ASSERT_ALWAYS(a % zb == za % zb);
    }
    cxx_mpz r = zb;
    gmp_auxx::mpz_sub(r, a, r);
    ASSERT_ALWAYS(r == za - zb);
}

template<typename T>
static void check_left_operand()
{
    for(long const b : { 0L, 1L, -1L, 17L, -17L,
            std::numeric_limits<long>::max(),
            std::numeric_limits<long>::min() }) {
        check_left_operand<T>(0, b);
        check_left_operand<T>(5, b);
        check_left_operand<T>(std::numeric_limits<T>::max(), b);
        check_left_operand<T>(std::numeric_limits<T>::min(), b);
        if constexpr (std::is_signed_v<T>)
            check_left_operand<T>(-5, b);
    }
}

/* reading a cxx_mpz from a stream must do what reading a long does: same
 * characters consumed, same stream state, same value.
 */
struct read_result {
    std::ios_base::iostate state;
    std::string rest;
};

template<typename T>
static read_result read_one(std::string const & s, std::ios_base::fmtflags f, T & v)
{
    std::istringstream is(s);
    is.flags(f);
    is >> v;
    read_result r { is.rdstate(), {} };
    is.clear();
    r.rest.assign(std::istreambuf_iterator<char>(is), {});
    return r;
}

static void check_read_like_long(std::string const & s, std::ios_base::fmtflags f)
{
    long v = 42;
    cxx_mpz z = 42;
    auto const rl = read_one(s, f, v);
    auto const rz = read_one(s, f, z);
    if (rl.state == rz.state && rl.rest == rz.rest && z == v)
        return;
    throw cado::error("reading \"{}\" with flags {:#x}:"
            " long -> {} (state {}, rest \"{}\");"
            " cxx_mpz -> {} (state {}, rest \"{}\")",
            s, unsigned(f), v, unsigned(rl.state), rl.rest,
            z, unsigned(rz.state), rz.rest);
}

static void check_read_large(std::string const & s, std::ios_base::fmtflags f,
        char const * value, int base, std::string const & rest)
{
    cxx_mpz z, e;
    mpz_set_str(e, value, base);
    auto const rz = read_one(s, f, z);
    long v;
    auto const rl = read_one(s, f, v);
    /* a long overflows, but stops at the same place */
    ASSERT_ALWAYS(rl.state & std::ios_base::failbit);
    ASSERT_ALWAYS(rl.rest == rz.rest);
    ASSERT_ALWAYS(rz.state == (rl.state & ~std::ios_base::failbit));
    ASSERT_ALWAYS(z == e && rz.rest == rest);
}

static void check_read()
{
    std::string const alphabet = "019af8xX+- ;";
    std::vector<std::string> inputs { "" };
    for (size_t k = 0, n = 0; k < 4; k++) {
        size_t const n1 = inputs.size();
        for (; n < n1; n++)
            for (char const c: alphabet)
                inputs.push_back(inputs[n] + c);
    }
    for (auto const basefield: { std::ios_base::dec, std::ios_base::hex,
            std::ios_base::oct, std::ios_base::fmtflags() }) {
        for (auto const ws: { std::ios_base::skipws, std::ios_base::fmtflags() }) {
            for (auto const & s: inputs)
                check_read_like_long(s, basefield | ws);
        }
    }

    auto const dec = std::ios_base::dec | std::ios_base::skipws;
    auto const hex = std::ios_base::hex | std::ios_base::skipws;
    auto const oct = std::ios_base::oct | std::ios_base::skipws;
    auto const none = std::ios_base::skipws;
    check_read_large(" 123456789012345678901234567890), (",
            dec, "123456789012345678901234567890", 10, "), (");
    check_read_large("-123456789012345678901234567890;",
            dec, "-123456789012345678901234567890", 10, ";");
    check_read_large("0123456789012345678901234567890",
            dec, "123456789012345678901234567890", 10, "");
    check_read_large("0x123456789abcdef0123456789abcdefg",
            hex, "123456789abcdef0123456789abcdef", 16, "g");
    check_read_large("123456789abcdef0123456789abcdef ",
            hex, "123456789abcdef0123456789abcdef", 16, " ");
    check_read_large("-0x123456789abcdef0123456789abcdef;",
            none, "-123456789abcdef0123456789abcdef", 16, ";");
    check_read_large("012345670123456701234567012345678",
            oct, "1234567012345670123456701234567", 8, "8");
}

int main()
{
    {
        const long z = 1234;
        checkcheck(z, "{}", "1234");
        checkcheck(z, "{:x}", "4d2");
        checkcheck(z, "{:#x}", "0x4d2");
        checkcheck(z, "{:X}", "4D2");
        checkcheck(z, "{:#X}", "0X4D2");
        checkcheck(-z, "{:#X}", "-0X4D2");
        checkcheck(-z, "{:#o}", "-02322");
    }
    check_left_operand<int>();
    check_left_operand<long>();
    check_left_operand<long long>();
    check_left_operand<int64_t>();
    check_left_operand<unsigned int>();
    check_left_operand<unsigned long>();
    check_left_operand<uint64_t>();
    check_read();
}
