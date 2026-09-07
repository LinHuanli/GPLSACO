cmake_minimum_required(VERSION 3.24)
project(gp_faco LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

option(GP_FACO_NATIVE "Build the external native FACO and diagnostic wrapper" ON)
option(GP_FACO_CUDA "Build CUDA scoring diagnostics" OFF)
option(GP_FACO_PYTHON "Build pybind11 diagnostic bindings" ON)

add_library(gp_faco_core STATIC cpp/src/program_cpu.cpp)
set_target_properties(gp_faco_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gp_faco_core PUBLIC cpp/include)
target_compile_options(gp_faco_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

if(GP_FACO_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gp_faco_cuda STATIC cuda/gp_score.cu)
    set_target_properties(gp_faco_cuda PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    target_include_directories(gp_faco_cuda PUBLIC cpp/include)
    target_compile_options(gp_faco_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(gp_faco_cuda PUBLIC gp_faco_core CUDA::cudart)
endif()

if(GP_FACO_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(gp_faco_ext cpp/src/bindings.cpp)
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
    file(SHA256 "${FACO_SOURCE_DIR}/src/local_search.cpp" GPFACO_REFERENCE_LS_HASH)
    target_compile_definitions(native_faco_diagnostic PRIVATE
        GPFACO_REFERENCE_LS_HASH="${GPFACO_REFERENCE_LS_HASH}")
    target_include_directories(native_faco_diagnostic PRIVATE "${FACO_SOURCE_DIR}/src")
    target_compile_options(native_faco_diagnostic PRIVATE -mavx2)
    target_link_libraries(native_faco_diagnostic PRIVATE native_faco_support)
endif()
