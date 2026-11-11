#include "cado.h" // IWYU pragma: keep

#include <cstdio>
#include <cstdlib>

#include <sstream>

#include "fmt/base.h"

#include "factorization_pattern.hpp"
#include "macros.h"

int main()
{
    factorization_pattern el1 {1000, { 1U, 2U, 3U} };
    factorization_pattern_list t {el1};
    if (!(el1 == t[0])) {
        fmt::print(stderr, "error with the test(1)!!!\n");
        return EXIT_FAILURE;
    }
    t.push_back(el1);
    // test realloc()
    const factorization_pattern el2 { 10000, { 11U, 10U, 9U, 8U, 7U }};
    t.push_back(el2);
    if (t[2] != el2) {
        fmt::print(stderr, "error with the test(2)!!!\n");
        return EXIT_FAILURE;
    }
    if (t[2] == el1) {
        fmt::print(stderr, "error with the test(3)!!!\n");
        return EXIT_FAILURE;
    }
    // set and get
    el1 = el2;
    if (el1 != el2) {
        fmt::print(stderr, "error with the test(4)!!!\n");
        return EXIT_FAILURE;
    }
    // round trip through the stream operators
    {
        std::ostringstream os;
        os << t;

        factorization_pattern_list t2;
        std::istringstream is(os.str());
        if (!(is >> t2)) {
            fmt::print(stderr, "read error on what we just wrote\n");
            return EXIT_FAILURE;
        }
        if (t2.size() != t.size()) {
            fmt::print(stderr, "error with the test(5)!!!\n");
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < t.size(); i++) {
            if (t2[i] != t[i]) {
                fmt::print(stderr, "error with the test(5)!!!\n");
                return EXIT_FAILURE;
            }
        }
        /* nb_elem is part of what a factorization_pattern compares as */
        t2[1].count = 21;
        if (t2[1] == t[1]) {
            fmt::print(stderr, "error with the test(6)!!!\n");
            return EXIT_FAILURE;
        }
    }

    /* comments are skipped, so that a file with the section labels that
     * the pinned test fixtures carry is still valid input */
    {
        std::istringstream is("# decomp_1024_21\n"
                              "[ 11 11 ]: 3844.62\n"
                              "# decomp_1024_22\n"
                              "[ 11 11 ]: 5798.8\n"
                              "[ 11 12 ]: 14024.1\n");
        factorization_pattern_list t4;
        if (!(is >> t4) || t4.size() != 3) {
            fmt::print(stderr, "error with the test(8)!!!\n");
            return EXIT_FAILURE;
        }
    }

    /* a malformed record must fail the stream, not abort */
    {
        std::istringstream is("[ 1 2 3 ; 1000\n");
        factorization_pattern_list t3;
        if (is >> t3) {
            fmt::print(stderr, "error with the test(7)!!!\n");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
