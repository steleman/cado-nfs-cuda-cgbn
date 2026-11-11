#include "cado.h" // IWYU pragma: keep
#include "gmp_mul_hook.hpp"

cado_mpz_mul_fn cado_mpz_mul_hook = nullptr;
size_t cado_mpz_mul_hook_threshold = (size_t) -1;
