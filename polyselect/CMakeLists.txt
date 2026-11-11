# defining a library avoids compiling the same object file once for each
# binary. The choice of setting it as a shared library is completely
# arbitrary here. A static would do as well.

add_library(polyselect_common
    ${CADO_SHARED}
    auxiliary.cpp
    murphyE.cpp
    area.cpp
    size_optimization.cpp
    sopt_gpu.cpp
    polyselect_alpha.cpp
    polyselect_norms.cpp
    )
target_link_libraries (polyselect_common PUBLIC utils)
cado_apply_pch(polyselect_common)
if(HAVE_CUDA)
    # GPU size optimization (size_optimization.cpp): the double
    # arithmetic must be the CPU's, hence no contraction into FMA
    add_library(sopt_gpu_kernels STATIC sopt_gpu_kernels.cu)
    # cado_gpu_claim (utils/gpu_claim.cpp)
    target_link_libraries(sopt_gpu_kernels PUBLIC utils)
    set_target_properties(sopt_gpu_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(sopt_gpu_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false -lineinfo>)
    target_link_libraries(sopt_gpu_kernels PUBLIC CUDA::cudart)
    target_link_libraries(polyselect_common PUBLIC sopt_gpu_kernels)
    add_executable(sopt_gpu_test EXCLUDE_FROM_ALL sopt_gpu_test.cpp)
    target_link_libraries(sopt_gpu_test PRIVATE polyselect_common)
endif()

# It _seems_ that several polyselect-like libraries share the same
# general working principle, and that these objects can be treated as a
# bundle. (we want them to be compiled only once, for efficiency)
add_library(polyselect_middle
    OBJECT
    polyselect_arith.cpp
    polyselect_collisions.cpp
    polyselect_data_series.c
    polyselect_special_q.c
    polyselect_thread.cpp
    polyselect_thread_team.cpp
    polyselect_thread_league.cpp
    polyselect_main_data.cpp
    polyselect_main_queue.cpp
    polyselect_match.cpp
    polyselect_poly_header.c
    polyselect_primes_table.c
    polyselect_priority_queue.c
    polyselect_proots.cpp
    polyselect_qroots.cpp
    polyselect_shash.cpp
    polyselect_stats.c
)
target_link_libraries(polyselect_middle PUBLIC polyselect_common ${pthread_libs})
if(HAVE_CUDA)
    # GPU collision test (polyselect_collisions.cpp)
    add_library(polyselect_gpu_kernels STATIC polyselect_gpu_kernels.cu)
    # cado_gpu_claim (utils/gpu_claim.cpp)
    target_link_libraries(polyselect_gpu_kernels PUBLIC utils)
    set_target_properties(polyselect_gpu_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(polyselect_gpu_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_link_libraries(polyselect_gpu_kernels PUBLIC CUDA::cudart)
    target_link_libraries(polyselect_middle PUBLIC polyselect_gpu_kernels)
endif()
mark_targets_as_openmp(polyselect_middle)
cado_apply_pch(polyselect_middle)

add_executable (polyselect
    polyselect.cpp
    )
target_link_libraries (polyselect PRIVATE polyselect_middle)
mark_targets_as_openmp(polyselect)
cado_apply_pch(polyselect)

add_executable (polyselect_ropt
    ropt_main.cpp
    best_polynomials_queue.cpp
    ropt_single_sublattice_priority_queue.cpp
    ropt_sublattice_priority_queue.cpp
    ropt_sublattice_crt.cpp
    ropt.cpp
    ropt_arith.cpp
    ropt_io.cpp
    ropt_linear.cpp
    ropt_param.cpp
    ropt_quadratic.cpp
    ropt_stage1.cpp
    ropt_stage2.cpp
    ropt_str.cpp
    ropt_tree.c
    ropt_sublattice_priority_queue.cpp
    ropt_gpu.cpp
)
target_link_libraries (polyselect_ropt PRIVATE polyselect_common ${pthread_libs})
if(HAVE_CUDA)
    # GPU root sieve (ropt_gpu.cpp)
    add_library(ropt_gpu_kernels STATIC ropt_gpu_kernels.cu)
    # cado_gpu_claim (utils/gpu_claim.cpp)
    target_link_libraries(ropt_gpu_kernels PUBLIC utils)
    set_target_properties(ropt_gpu_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(ropt_gpu_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_link_libraries(ropt_gpu_kernels PUBLIC CUDA::cudart)
    target_link_libraries(polyselect_ropt PRIVATE ropt_gpu_kernels)
endif()
mark_targets_as_openmp(polyselect_ropt)
cado_apply_pch(polyselect_ropt)


add_executable (polyselect_gfpn polyselect_gfpn.cpp gfpkdlpolyselect.cpp)
cado_apply_pch(polyselect_gfpn)
target_link_libraries (polyselect_gfpn PRIVATE  polyselect_common)

add_executable (polyselect3 polyselect3.cpp)
mark_targets_as_openmp(polyselect3)
cado_apply_pch(polyselect3)
target_link_libraries (polyselect3 PRIVATE polyselect_common)

if(ENABLE_SHARED)
    install(TARGETS polyselect_common DESTINATION ${LIBSUFFIX}/polyselect)
endif()
install(TARGETS polyselect RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)
install(TARGETS polyselect_ropt RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)
install(TARGETS polyselect3 RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)

# dlpolyselect
add_executable (dlpolyselect dlpolyselect.cpp)
mark_targets_as_openmp(dlpolyselect)
cado_apply_pch(dlpolyselect)
target_link_libraries (dlpolyselect PRIVATE polyselect_common)
install(TARGETS dlpolyselect RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)


include(CMakeLists-nodist.txt OPTIONAL)

# skewness
add_executable (skewness skewness.cpp)
cado_apply_pch(skewness)
target_link_libraries (skewness PRIVATE polyselect_common)
install(TARGETS skewness RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)

# score
add_executable (score score.cpp)
cado_apply_pch(score)
target_link_libraries (score PRIVATE polyselect_common)
install(TARGETS score RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)

# alpha
add_executable (alpha alpha.cpp)
cado_apply_pch(alpha)
target_link_libraries (alpha PRIVATE polyselect_common)
install(TARGETS alpha RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)
