include_directories (utils_st)
include_directories (../ecm)
include_directories (${CMAKE_CURRENT_BINARY_DIR})

set (SUF_UTILS utils_st/)
set (SRC_UTILS
    ${SUF_UTILS}convex_hull.cpp ${SUF_UTILS}cofactoring_method.cpp
    ${SUF_UTILS}cofactoring_strategy.cpp)

# The strategy builder reads factorization patterns but does not generate
# them; only list_factorization_patterns does.
set (SRC_PATTERN_IO ${SUF_UTILS}factorization_pattern.cpp)
set (SRC_PATTERNS factorization_patterns.cpp ${SRC_PATTERN_IO})

set (LIB_EXEC facul utils)

add_executable(survey_cofactoring_methods survey_cofactoring_methods.cpp cofactoring_method_bench.cpp ${SRC_UTILS})
target_link_libraries(survey_cofactoring_methods PRIVATE ${LIB_EXEC})

add_executable(bench_cofactoring_methods bench_cofactoring_methods.cpp cofactoring_method_bench.cpp ${SRC_UTILS})
target_link_libraries(bench_cofactoring_methods PRIVATE ${LIB_EXEC})

add_executable(build_cofactoring_strategies
  build_cofactoring_strategies.cpp
  build_strategies.cpp
  ${SRC_PATTERN_IO}
  ${SRC_UTILS})

target_link_libraries(build_cofactoring_strategies PRIVATE ${LIB_EXEC})

add_executable(choose_cofactoring_strategies
  choose_cofactoring_strategies.cpp
  select_best_strategies.cpp
  ${SRC_UTILS})

target_link_libraries(choose_cofactoring_strategies PRIVATE ${LIB_EXEC})

add_executable(replay_cofactoring replay_cofactoring.cpp)
target_link_libraries(replay_cofactoring PRIVATE ${LIB_EXEC})

add_executable(cofactoring_trajectories cofactoring_trajectories.cpp ${SRC_PATTERNS})
target_link_libraries(cofactoring_trajectories PRIVATE ${LIB_EXEC})

add_executable(strategies_from_trajectories strategies_from_trajectories.cpp)
target_link_libraries(strategies_from_trajectories PRIVATE ${LIB_EXEC})

add_executable(refine_strategy_file refine_strategy_file.cpp)
target_link_libraries(refine_strategy_file PRIVATE ${LIB_EXEC})

add_executable(list_factorization_patterns
  list_factorization_patterns.cpp
  ${SRC_PATTERNS})

target_link_libraries(list_factorization_patterns PRIVATE utils)


include(CMakeLists-nodist.txt OPTIONAL)
