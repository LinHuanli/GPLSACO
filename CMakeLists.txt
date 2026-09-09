cmake_minimum_required(VERSION 3.24)
project(gp_faco LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

option(GP_FACO_NATIVE "Build the external native FACO and diagnostic wrapper" ON)
option(GP_FACO_CUDA "Build CUDA solver and diagnostic targets" OFF)
option(GP_FACO_PYTHON "Build pybind11 solver and scoring bindings" ON)
set(GP_FACO_SEARCH_NUMERIC "exact" CACHE STRING "Search arithmetic: exact, fp32, fp32_fast")
set_property(CACHE GP_FACO_SEARCH_NUMERIC PROPERTY STRINGS exact fp32 fp32_fast)
set(GP_FACO_WARPS_PER_BLOCK "2" CACHE STRING "Ant warps per production block: 2, 4, 8")
set_property(CACHE GP_FACO_WARPS_PER_BLOCK PROPERTY STRINGS 2 4 8)
option(GP_FACO_PARALLEL_LS_PREFIX "Parallel unrestricted warp LS candidate prefix" ON)
option(GP_FACO_SHARED_ANT_FLAGS "Keep warp ant visited/queued flags in shared memory" ON)
if(NOT GP_FACO_WARPS_PER_BLOCK MATCHES "^(2|4|8)$")
    message(FATAL_ERROR "Ant warps per block must be 2, 4 or 8")
endif()
if(NOT GP_FACO_SEARCH_NUMERIC MATCHES "^(exact|fp32|fp32_fast)$")
    message(FATAL_ERROR "Unknown search arithmetic profile")
endif()

add_library(gp_faco_core STATIC cpp/src/program_cpu.cpp cpp/src/faco_cpu.cpp cpp/src/prepared_problem.cpp cpp/src/sparse_graph.cpp cpp/src/control_cpu.cpp cpp/src/counted_state.cpp)
set_target_properties(gp_faco_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gp_faco_core PUBLIC cpp/include)
target_compile_options(gp_faco_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
add_executable(deadline_semantics tests/cpp/test_deadline.cpp)
target_link_libraries(deadline_semantics PRIVATE gp_faco_core)
add_test(NAME deadline_semantics COMMAND deadline_semantics)
add_executable(hard_edge_semantics tests/cpp/test_hard_edges.cpp)
target_link_libraries(hard_edge_semantics PRIVATE gp_faco_core)
add_test(NAME hard_edge_semantics COMMAND hard_edge_semantics
    "${CMAKE_CURRENT_BINARY_DIR}/hard_edge_results.json")
add_executable(escape_edge_semantics tests/cpp/test_escape_edges.cpp)
target_link_libraries(escape_edge_semantics PRIVATE gp_faco_core)
add_test(NAME escape_edge_semantics COMMAND escape_edge_semantics
    "${CMAKE_CURRENT_BINARY_DIR}/escape_edge_results.json")
add_executable(escape_slot_semantics tests/cpp/test_escape_slots.cpp)
target_link_libraries(escape_slot_semantics PRIVATE gp_faco_core)
add_test(NAME escape_slot_semantics COMMAND escape_slot_semantics)
add_executable(control_cpu_semantics tests/cpp/test_control_cpu.cpp)
target_link_libraries(control_cpu_semantics PRIVATE gp_faco_core)
add_test(NAME control_cpu_semantics COMMAND control_cpu_semantics
    "${CMAKE_CURRENT_BINARY_DIR}/control_cpu_results.json")
add_executable(baseline_policy_semantics tests/cpp/test_baseline_policy.cpp)
target_link_libraries(baseline_policy_semantics PRIVATE gp_faco_core)
add_test(NAME baseline_policy_semantics COMMAND baseline_policy_semantics
    "${CMAKE_CURRENT_BINARY_DIR}/baseline_policy_results.json")
add_executable(graph_preparation_semantics tests/cpp/test_graph_preparation.cpp)
target_link_libraries(graph_preparation_semantics PRIVATE gp_faco_core)
add_test(NAME graph_preparation_semantics COMMAND graph_preparation_semantics
    "${CMAKE_CURRENT_BINARY_DIR}/graph_preparation_results.json")

if(GP_FACO_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gp_faco_cuda STATIC cuda/gp_score.cu cuda/faco_operations.cu cuda/fixed_faco.cu cuda/batch_engine.cu cuda/hard_diagnostic.cu)
    set_target_properties(gp_faco_cuda PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    target_include_directories(gp_faco_cuda PUBLIC cpp/include)
    target_compile_definitions(gp_faco_cuda PRIVATE
        GPFACO_WARPS_PER_BLOCK=${GP_FACO_WARPS_PER_BLOCK}
        GPFACO_PARALLEL_LS_PREFIX=$<BOOL:${GP_FACO_PARALLEL_LS_PREFIX}>
        GPFACO_SHARED_ANT_FLAGS=$<BOOL:${GP_FACO_SHARED_ANT_FLAGS}>)
    target_compile_options(gp_faco_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--generate-line-info>)
    if(NOT GP_FACO_SEARCH_NUMERIC STREQUAL "exact")
        target_compile_definitions(gp_faco_cuda PRIVATE GPFACO_SEARCH_FP32=1)
    endif()
    if(GP_FACO_SEARCH_NUMERIC STREQUAL "fp32_fast")
        target_compile_options(gp_faco_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    target_link_libraries(gp_faco_cuda PUBLIC gp_faco_core CUDA::cudart)
    add_executable(faco_cuda_semantics tests/cpp/test_cuda_faco.cpp)
    target_compile_options(faco_cuda_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(faco_cuda_semantics PRIVATE gp_faco_cuda)
    add_test(NAME faco_cuda_semantics COMMAND faco_cuda_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/cuda_faco_results.json")
    add_executable(fixed_faco_semantics tests/cpp/test_fixed_faco.cpp)
    target_compile_options(fixed_faco_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(fixed_faco_semantics PRIVATE gp_faco_cuda)
    add_test(NAME fixed_faco_semantics COMMAND fixed_faco_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/fixed_faco_results.json")
    add_executable(batch_engine_semantics tests/cpp/test_batch_engine.cpp)
    target_compile_options(batch_engine_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(batch_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME batch_engine_semantics COMMAND batch_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/batch_engine_results.json")
    add_executable(control_engine_semantics tests/cpp/test_control_engine.cpp)
    target_compile_options(control_engine_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(control_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME control_engine_semantics COMMAND control_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/control_engine_results.json")
    add_executable(hard_cuda_semantics tests/cpp/test_hard_cuda.cpp)
    target_compile_options(hard_cuda_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(hard_cuda_semantics PRIVATE gp_faco_cuda)
    add_test(NAME hard_cuda_semantics COMMAND hard_cuda_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/hard_cuda_results.json")
    add_executable(profiling_semantics tests/cpp/test_profiling.cpp)
    target_compile_options(profiling_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(profiling_semantics PRIVATE gp_faco_cuda)
    add_test(NAME profiling_semantics COMMAND profiling_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/profiling_semantics.json")
    add_executable(evaluation_count_semantics tests/cpp/test_evaluation_count.cpp)
    target_compile_options(evaluation_count_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(evaluation_count_semantics PRIVATE gp_faco_cuda)
    add_test(NAME evaluation_count_semantics COMMAND evaluation_count_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/evaluation_count_semantics.json")
    add_executable(baseline_engine_semantics tests/cpp/test_baseline_engine.cpp)
    target_compile_options(baseline_engine_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(baseline_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME baseline_engine_semantics COMMAND baseline_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/baseline_engine_results.json")
    add_executable(factorial_engine_semantics tests/cpp/test_factorial_engine.cpp)
    target_compile_options(factorial_engine_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(factorial_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME factorial_engine_semantics COMMAND factorial_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/factorial_engine_results.json")
    add_executable(behavior_engine_semantics tests/cpp/test_behavior_engine.cpp)
    target_compile_options(behavior_engine_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(behavior_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME behavior_engine_semantics COMMAND behavior_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/behavior_engine_results.json")
    add_executable(hard_engine_semantics tests/cpp/test_hard_engine.cpp)
    target_link_libraries(hard_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME hard_engine_semantics COMMAND hard_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/hard_engine_results.json")
    add_executable(escape_engine_semantics tests/cpp/test_escape_engine.cpp tests/cpp/escape_primitives.cu)
    set_target_properties(escape_engine_semantics PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(escape_engine_semantics PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--generate-line-info>)
    target_link_libraries(escape_engine_semantics PRIVATE gp_faco_cuda)
    add_test(NAME escape_engine_semantics COMMAND escape_engine_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/escape_engine_results.json")
    add_executable(state_fork_semantics tests/cpp/test_state_fork.cpp)
    target_compile_options(state_fork_semantics PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_link_libraries(state_fork_semantics PRIVATE gp_faco_cuda)
    add_test(NAME state_fork_semantics COMMAND state_fork_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/state_fork_results.json")
endif()

if(GP_FACO_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(gp_faco_ext cpp/src/bindings.cpp)
    target_compile_definitions(gp_faco_ext PRIVATE GPFACO_NUMERIC_BACKEND="${GP_FACO_SEARCH_NUMERIC}")
    target_link_libraries(gp_faco_ext PRIVATE gp_faco_core)
    if(GP_FACO_CUDA)
        target_compile_definitions(gp_faco_ext PRIVATE GPFACO_CUDA=1)
        target_link_libraries(gp_faco_ext PRIVATE gp_faco_cuda)
    endif()
endif()

if(GP_FACO_NATIVE)
    set(FACO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../references/ACO-TSP-Adaptive-Tuning"
        CACHE PATH "Read-only native FACO source snapshot")
    find_package(OpenMP REQUIRED)
    set(FACO_SUPPORT_SOURCES
        "${FACO_SOURCE_DIR}/src/problem_instance.cpp"
        "${FACO_SOURCE_DIR}/src/local_search.cpp"
        "${FACO_SOURCE_DIR}/src/utils.cpp"
        "${FACO_SOURCE_DIR}/src/rand.cpp"
        "${FACO_SOURCE_DIR}/src/progargs.cpp")
    add_library(native_faco_support OBJECT ${FACO_SUPPORT_SOURCES})
    target_link_libraries(native_faco_support PUBLIC OpenMP::OpenMP_CXX)
    # 上游随机数/距离代码使用 AVX2；只应用于 CPU 参考目标。
    # GCC 15 不再间接提供这些整数类型；命令行补头，不改只读上游源码。
    target_compile_options(native_faco_support PRIVATE -mavx2 -include cstdint)
    add_executable(native_faco "${FACO_SOURCE_DIR}/src/faco.cpp")
    target_compile_options(native_faco PRIVATE -mavx2)
    target_link_libraries(native_faco PRIVATE native_faco_support)
    add_executable(native_faco_diagnostic cpp/src/native_diagnostic.cpp)
    target_include_directories(native_faco_diagnostic PRIVATE "${FACO_SOURCE_DIR}/src")
    target_compile_options(native_faco_diagnostic PRIVATE -mavx2)
    target_link_libraries(native_faco_diagnostic PRIVATE native_faco_support)
    add_executable(faco_native_semantics tests/cpp/test_native_semantics.cpp)
    target_include_directories(faco_native_semantics PRIVATE "${FACO_SOURCE_DIR}/src")
    target_compile_options(faco_native_semantics PRIVATE -mavx2 -ffp-contract=off)
    target_link_libraries(faco_native_semantics PRIVATE native_faco_support gp_faco_core)
    add_test(NAME faco_native_semantics COMMAND faco_native_semantics
        "${CMAKE_CURRENT_BINARY_DIR}/native_semantics_results.json")
endif()

# 2022 原始 FACO 独立编译，避免把节点重定位版本误称为原始算法。
option(GP_FACO_2022 "Build the original 2022 FACO comparator" ON)
option(GP_FACO_2022_CONTINUOUS "Use the named continuous-distance adaptation" OFF)
if(GP_FACO_2022)
    set(FACO_2022_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../references/FocusedACO"
        CACHE PATH "Read-only original 2022 FACO source")
    find_package(OpenMP REQUIRED)
    if(GP_FACO_2022_CONTINUOUS AND NOT EXISTS "${FACO_2022_SOURCE_DIR}/adaptation.json")
        message(FATAL_ERROR "Prepare the named continuous adaptation and set FACO_2022_SOURCE_DIR")
    endif()
    add_library(faco_2022_support OBJECT
        "${FACO_2022_SOURCE_DIR}/src/problem_instance.cpp"
        "${FACO_2022_SOURCE_DIR}/src/local_search.cpp"
        "${FACO_2022_SOURCE_DIR}/src/utils.cpp"
        "${FACO_2022_SOURCE_DIR}/src/rand.cpp"
        "${FACO_2022_SOURCE_DIR}/src/progargs.cpp")
    target_link_libraries(faco_2022_support PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(faco_2022_support PRIVATE -mavx2 -include cstdint)
    add_executable(faco_2022 cpp/src/faco_2022.cpp)
    if(GP_FACO_2022_CONTINUOUS)
        target_compile_definitions(faco_2022 PRIVATE GPFACO_CONTINUOUS_ADAPTATION=1)
    endif()
    target_include_directories(faco_2022 PRIVATE "${FACO_2022_SOURCE_DIR}/src")
    target_compile_options(faco_2022 PRIVATE -mavx2 -include cstdint)
    target_link_libraries(faco_2022 PRIVATE faco_2022_support)
endif()
