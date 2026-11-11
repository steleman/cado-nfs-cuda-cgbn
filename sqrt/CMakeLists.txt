
if (${WITH_MPI})
add_definitions(-DWITH_MPI)
endif()

add_executable(sqrt sqrt.cpp sqrt_gpu.cpp)
target_link_libraries(sqrt PRIVATE utils_with_io)
mark_targets_as_openmp(sqrt)
if(HAVE_CUDA)
    # GPU multiplication of large integers (sqrt_gpu.cpp)
    add_library(sqrt_gpu_mul STATIC sqrt_gpu_mul.cu)
    # cado_gpu_claim (utils/gpu_claim.cpp)
    target_link_libraries(sqrt_gpu_mul PUBLIC utils)
    set_target_properties(sqrt_gpu_mul PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(sqrt_gpu_mul PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_link_libraries(sqrt_gpu_mul PUBLIC CUDA::cudart)
    target_link_libraries(sqrt PRIVATE sqrt_gpu_mul)
    add_executable(sqrt_gpu_mul_test EXCLUDE_FROM_ALL sqrt_gpu_mul_test.cpp)
    target_link_libraries(sqrt_gpu_mul_test PRIVATE sqrt_gpu_mul ${gmp_libname})
endif()
install(TARGETS sqrt RUNTIME DESTINATION ${LIBSUFFIX}/sqrt)

add_executable(mgmsqrt mgmsqrt.cpp)
target_link_libraries(mgmsqrt PRIVATE utils_with_io)
mark_targets_as_openmp(mgmsqrt)
install(TARGETS mgmsqrt RUNTIME DESTINATION ${LIBSUFFIX}/sqrt)


add_executable(crtalgsqrt ${EXCLUDE_FROM_RELEASE_BUILD} crtalgsqrt.cpp
    powers_of_p.cpp ab_source.cpp knapsack.c
    sqrt_cachefiles.cpp)
target_link_libraries(crtalgsqrt PRIVATE utils)

mark_targets_as_mpi(crtalgsqrt)

include(CMakeLists-nodist.txt OPTIONAL)
