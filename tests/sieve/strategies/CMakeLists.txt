set (SUF_SRC ${CADO_NFS_SOURCE_DIR}/sieve/strategies)

set (SUF_UTILS ${SUF_SRC}/utils_st/)
set (SRC_UTILS
  ${SUF_UTILS}convex_hull.cpp ${SUF_UTILS}cofactoring_method.cpp
  ${SUF_UTILS}cofactoring_strategy.cpp ${SUF_UTILS}factorization_pattern.cpp)

set (LIB_EXEC m facul utils)

include_directories(${CADO_NFS_SOURCE_DIR})
include_directories(${CADO_NFS_SOURCE_DIR}/sieve)
include_directories(${CADO_NFS_SOURCE_DIR}/sieve/ecm)
include_directories(${CADO_NFS_SOURCE_DIR}/sieve/strategies)
include_directories(${CADO_NFS_SOURCE_DIR}/sieve/strategies/utils_st)

cado_define_test(
  test_factorization_patterns.cpp
  ${SUF_SRC}/factorization_patterns.cpp
  ${SUF_UTILS}/factorization_pattern.cpp
  ARGUMENTS 60 800000
  LIBRARIES ${LIB_EXEC})

if (HAVE_REGEX_H)
cado_define_test(test_generate_composite_numbers.cpp
  ${SUF_SRC}/cofactoring_method_bench.cpp ${SRC_UTILS}
  LIBRARIES ${LIB_EXEC})

cado_define_test(test_generate_strategies.cpp
  ${SUF_SRC}/cofactoring_method_bench.cpp
  ${SUF_SRC}/build_strategies.cpp
  ${SUF_SRC}/factorization_patterns.cpp
  ${SRC_UTILS}
  LIBRARIES ${LIB_EXEC}
  TIMEOUT 80)

cado_define_test(test_filtering.cpp
  ${SUF_SRC}/cofactoring_method_bench.cpp
  ${SRC_UTILS}
  LIBRARIES ${LIB_EXEC})

cado_define_test(test_benchfm.cpp
  ${SUF_SRC}/cofactoring_method_bench.cpp
  ${SRC_UTILS}
  LIBRARIES ${LIB_EXEC})

cado_define_test(SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_missing_method_family.sh
    PROVIDE_TEMPORARY_WDIR
    TARGET_DEPENDENCIES build_cofactoring_strategies
        list_factorization_patterns
    TIMEOUT 40
)

# build_cofactoring_strategies and choose_cofactoring_strategies are pure functions of their input files, so their output
# is pinned byte for byte. survey_cofactoring_methods and bench_cofactoring_methods measure wall-clock time, so
# theirs is pinned with the timing columns stripped. Both scripts take
# REGEN=1 to rewrite their references in the source tree.
#
# Both are guarded, because a byte-exact reference is only reproducible
# where the arithmetic that produced it is. See the README for the
# details; in short, this pipeline chooses between candidate strategies
# by comparing a (probability, time) trade-off, exact ties are common,
# and a compiler that keeps intermediate doubles in wider registers
# breaks those ties the other way. It then picks strategies that are just
# as good but not the same ones, which no tolerance in the comparison can
# absorb.
if(HAVE_IEEE_DOUBLE_EVALUATION)
cado_define_test(SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_strategies_pinned.sh
    PROVIDE_TEMPORARY_WDIR
    TARGET_DEPENDENCIES build_cofactoring_strategies choose_cofactoring_strategies
        list_factorization_patterns
    TIMEOUT 40
)
endif()

# This one needs the word size as well: survey_cofactoring_methods picks
# the sizes of its random composites with random_uniform(), which draws
# ULONG_MAX worth of randomness per call, so the seeded stream itself --
# and hence the methods surveyed -- depends on how wide an unsigned long
# is.
if(HAVE_IEEE_DOUBLE_EVALUATION AND ULONG_BITS EQUAL 64)
cado_define_test(SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_strategies_pinned_notime.sh
    PROVIDE_TEMPORARY_WDIR
    TARGET_DEPENDENCIES survey_cofactoring_methods bench_cofactoring_methods
    TIMEOUT 90
)
endif()

if(DEFINED ENV{CHECKS_EXPENSIVE})
    cado_define_test(SCRIPT
        ${CMAKE_CURRENT_SOURCE_DIR}/test_main_script.sh
        PROVIDE_TEMPORARY_WDIR
        TARGET_DEPENDENCIES survey_cofactoring_methods bench_cofactoring_methods
            build_cofactoring_strategies choose_cofactoring_strategies
            list_factorization_patterns las makefb
        TIMEOUT 240
    )
endif()
endif()

add_subdirectory (utils_st)
