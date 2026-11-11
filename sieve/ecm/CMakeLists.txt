
# All executables are somewhat optional ; testbench is the one which is
# used for tests.

# Would compile, but implies a dependency on gmp-ecm, or so it seems.
# add_executable(find_primes_hit find_primes_hit.c)
# target_link_libraries(find_primes_hit utils)


# macro(variant_facul base typewidth)
# string(TOUPPER ${typewidth} TW)
#     SET(src ${CMAKE_CURRENT_SOURCE_DIR}/${base}.c)
#     SET(dst ${CMAKE_CURRENT_BINARY_DIR}/${base}_${typewidth}.c)
# add_custom_command(OUTPUT  ${dst}
#     COMMAND ln -s ${src} ${dst}
#     DEPENDS  ${src})
# set_source_files_properties(${dst} PROPERTIES COMPILE_FLAGS -DMODREDC${TW})
# endmacro()
# 
# variant_facul(pm1 15ul)
# variant_facul(pp1 15ul)
# variant_facul(ecm 15ul)
# variant_facul(facul_doit 15ul)
# variant_facul(pm1 ul)
# variant_facul(pp1 ul)
# variant_facul(ecm ul)
# variant_facul(facul_doit ul)

include_directories(. ..)

# We now have wrappers, so it's very easy from the cmake perspective.
add_library(facul
    OBJECT
        facul_doit.cpp
        stage2.cpp
        bytecode.c
        pm1.cpp
        pp1.cpp
        ecm.cpp
        makeplan.cpp facul.cpp modset.cpp
        facul_method.cpp
        facul_strategies_stats.cpp
        facul_strategies.cpp
        mpqs_doit.cpp
        batch.cpp
        ec_arith_Weierstrass_new.cpp)

mark_targets_as_openmp(facul)

target_link_libraries(facul PUBLIC utils ${gmp_libname})

add_executable(testbench testbench.cpp)
target_link_libraries(testbench PRIVATE facul utils)


add_executable(finishbatch finishbatch.cpp)
target_link_libraries(finishbatch PRIVATE las_core_0 las_core_a facul utils)
mark_targets_as_openmp(finishbatch)

add_executable(finishecm finishecm.cpp)
target_link_libraries(finishecm PRIVATE facul utils)

add_executable(precompbatch precompbatch.cpp)
target_link_libraries(precompbatch PRIVATE las_core_0 facul utils)
mark_targets_as_openmp(precompbatch)

include(CMakeLists-nodist.txt OPTIONAL)
