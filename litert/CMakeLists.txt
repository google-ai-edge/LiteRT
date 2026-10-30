# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL
    "Suppress deprecation warnings from vendored CMake dependencies" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL
    "Suppress developer warnings from vendored CMake dependencies")
set(ENV{PYTHONWARNINGS} "ignore::DeprecationWarning")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress deprecation warnings (matching Bazel build behavior in .bazelrc)
# The codebase has deprecated Error constructors that are still in use.
# These are GCC/Clang spellings; an MSVC-style driver (cl.exe, and clang-cl,
# which CMake also reports as MSVC) rejects them outright -- cl.exe fails with
# "D8021: invalid numeric argument '/Wno-deprecated-declarations'".
if(MSVC)
  add_compile_options(
    /wd4996  # deprecated declarations
    /wd4715  # not all control paths return a value
  )
else()
  add_compile_options(
    -Wno-deprecated-declarations
    -Wno-nullability-completeness
    -Wno-return-type
    -fpermissive
  )
endif()
add_compile_definitions(CL_TARGET_OPENCL_VERSION=300)

if(WIN32)
  # TFLite sets these for its own targets (tflite/CMakeLists.txt); LiteRT's
  # targets include the same headers and need them too. Without NOMINMAX the
  # windows.h min/max macros break every generated FlatBuffers accessor named
  # min() or max(), e.g. tflite/converter/schema/schema_generated.h.
  add_compile_definitions(
    NOMINMAX              # keep windows.h from defining min()/max()
    WIN32_LEAN_AND_MEAN
    NOGDI                 # windows.h ERROR macro vs. logging severity names
    _USE_MATH_DEFINES     # M_PI and friends
    _CRT_SECURE_NO_WARNINGS
  )
endif()

# Cmake flattens the transitive public and interface link to emit the final link
# command. Which will trigger a warning on Apple platforms (ld64 still dedups
# those archives correctly) For more details, see
# https://gitlab.kitware.com/cmake/cmake/-/issues/25297
if(APPLE)
  add_link_options(-Wl,-no_warn_duplicate_libraries)
endif()

if(CMAKE_SYSTEM_PROCESSOR)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LITERT_SYSTEM_PROCESSOR_LOWER)
else()
  set(LITERT_SYSTEM_PROCESSOR_LOWER "")
endif()
set(_litert_disable_kleidiai_default OFF)
if(LITERT_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|i[3-6]86)$")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_litert_disable_kleidiai_default ON)
  endif()
endif()

# Options
option(LITERT_BUILD_TESTS "Build LiteRT tests" OFF)
option(LITERT_AUTO_BUILD_TFLITE "Automatically build TFLite if not found" ON)
option(LITERT_ENABLE_GPU "Enable GPU acceleration support" ON)
option(LITERT_ENABLE_NPU "Enable NPU acceleration support" ON)
option(LITERT_DISABLE_KLEIDIAI "Disable KleidiAI delegate when bundling host TFLite (default ON for Linux/macOS x86_64 hosts)" ${_litert_disable_kleidiai_default})
option(LITERT_BUILD_SUPPORT_LIBS "Enable build for the support libraries" OFF)
option(LITERT_ENABLE_UV_VENV "Enables uv virtual environment for the build" OFF)

set(LITERT_HOST_C_COMPILER "" CACHE FILEPATH "Host C compiler used for helper host-side tools (e.g. FlatBuffers flatc)")
set(LITERT_HOST_CXX_COMPILER "" CACHE FILEPATH "Host C++ compiler used for helper host-side tools (e.g. FlatBuffers flatc)")

set(LITERT_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
set(LITERT_BUILD_COMMON_GENERATED_DIR "${LITERT_GENERATED_INCLUDE_DIR}/litert/build_common")
file(MAKE_DIRECTORY "${LITERT_BUILD_COMMON_GENERATED_DIR}")

set(LITERT_BUILD_CONFIG_DISABLE_GPU 0)
if(NOT LITERT_ENABLE_GPU)
  set(LITERT_BUILD_CONFIG_DISABLE_GPU 1)
endif()

set(LITERT_BUILD_CONFIG_DISABLE_NPU 0)
if(NOT LITERT_ENABLE_NPU)
  set(LITERT_BUILD_CONFIG_DISABLE_NPU 1)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/build_common/build_config.h.in
    ${LITERT_BUILD_COMMON_GENERATED_DIR}/build_config.h
)

include_directories(${LITERT_GENERATED_INCLUDE_DIR})

# TFLite configuration
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite" CACHE PATH "Path to TFLite source directory")
set(TFLITE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/tflite_build" CACHE PATH "Path to TFLite build directory")

option(TFLITE_ENABLE_XNNPACK "Enable XNNPACK in TFLite" ON)
if(LITERT_DISABLE_KLEIDIAI)
  set(XNNPACK_ENABLE_KLEIDIAI OFF CACHE BOOL "Disable KleidiAI in XNNPACK")
endif()

# MSVC's cl.exe targeting ARM64 ships none of the ACLE headers or intrinsics
# these microkernel families need: <arm_fp16.h>, <arm_bf16.h> and <arm_sve.h>
# do not exist, and arm64_neon.h declares no bf16 intrinsics. XNNPACK selects
# them from the target ISA alone, so a cl.exe build fails to compile them.
# clang-cl has all of them, so only gate the MSVC frontend.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows"
   AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
   AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"
   AND LITERT_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$")
  message(STATUS
      "LiteRT: cl.exe on ARM64 -- disabling XNNPACK FP16/BF16/SVE2/SME "
      "microkernels (no ACLE support). Build with clang-cl to keep them.")
  foreach(_litert_xnn_off
          XNNPACK_ENABLE_ARM_FP16_SCALAR
          XNNPACK_ENABLE_ARM_FP16_VECTOR
          XNNPACK_ENABLE_ARM_BF16
          XNNPACK_ENABLE_ARM_SVE2
          XNNPACK_ENABLE_ARM_SME
          XNNPACK_ENABLE_ARM_SME2)
    set(${_litert_xnn_off} OFF CACHE BOOL "Unsupported by cl.exe on ARM64")
  endforeach()
endif()

if(CMAKE_CROSSCOMPILING)
  # Build or reuse a host flatc for cross compilation.
  if(NOT TFLITE_HOST_TOOLS_DIR)
    set(_root_host_flatc_dir "${CMAKE_CURRENT_SOURCE_DIR}/../host_flatc_build")
    if(EXISTS "${_root_host_flatc_dir}/_deps/flatbuffers-build/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
      set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build" CACHE PATH "Host tools directory" FORCE)
      message(STATUS "Reusing host flatc from ${TFLITE_HOST_TOOLS_DIR}")
    elseif(EXISTS "${_root_host_flatc_dir}/_deps/flatbuffers-build/Release/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
      set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build/Release" CACHE PATH "Host tools directory" FORCE)
      message(STATUS "Reusing host flatc from ${TFLITE_HOST_TOOLS_DIR}")
    else()
      file(MAKE_DIRECTORY "${_root_host_flatc_dir}")
      file(WRITE "${_root_host_flatc_dir}/CMakeLists.txt" "cmake_minimum_required(VERSION 3.16)\nproject(HostFlatc LANGUAGES C CXX)\ninclude(FetchContent)\nFetchContent_Declare(flatbuffers GIT_REPOSITORY https://github.com/google/flatbuffers.git GIT_TAG v25.9.23)\nset(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\nFetchContent_MakeAvailable(flatbuffers)\n")
      set(_litert_host_flatc_cmake_args -S . -B . -DCMAKE_BUILD_TYPE=Release)
      if(CMAKE_HOST_WIN32 AND CMAKE_CROSSCOMPILING AND NOT LITERT_HOST_C_COMPILER AND NOT LITERT_HOST_CXX_COMPILER)
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$" AND
           (NOT DEFINED ENV{CMAKE_GENERATOR} OR "$ENV{CMAKE_GENERATOR}" MATCHES "^Visual Studio"))
          list(APPEND _litert_host_flatc_cmake_args -A x64)
        endif()
      endif()
      set(_litert_host_c_compiler "${LITERT_HOST_C_COMPILER}")
      if(NOT _litert_host_c_compiler AND NOT CMAKE_HOST_WIN32)
        if(EXISTS "/usr/bin/clang")
          set(_litert_host_c_compiler "/usr/bin/clang")
        else()
          find_program(_litert_host_c_compiler NAMES clang cc gcc)
        endif()
      endif()
      if(_litert_host_c_compiler)
        list(APPEND _litert_host_flatc_cmake_args -DCMAKE_C_COMPILER=${_litert_host_c_compiler})
      endif()
      set(_litert_host_cxx_compiler "${LITERT_HOST_CXX_COMPILER}")
      if(NOT _litert_host_cxx_compiler AND NOT CMAKE_HOST_WIN32)
        if(EXISTS "/usr/bin/clang++")
          set(_litert_host_cxx_compiler "/usr/bin/clang++")
        else()
          find_program(_litert_host_cxx_compiler NAMES clang++ c++ g++)
        endif()
      endif()
      if(_litert_host_cxx_compiler)
        list(APPEND _litert_host_flatc_cmake_args -DCMAKE_CXX_COMPILER=${_litert_host_cxx_compiler})
      endif()
      execute_process(
        COMMAND ${CMAKE_COMMAND} ${_litert_host_flatc_cmake_args}
        WORKING_DIRECTORY "${_root_host_flatc_dir}"
        RESULT_VARIABLE HOSTCFG_RC
        ERROR_VARIABLE HOSTCFG_ERR)
      if(NOT HOSTCFG_RC EQUAL 0)
        message(FATAL_ERROR "Failed to configure host flatbuffers:\n${HOSTCFG_ERR}")
      endif()
      if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
        set(_flatc_jobs 4)
      else()
        set(_flatc_jobs ${CMAKE_BUILD_PARALLEL_LEVEL})
      endif()
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --target flatc --config Release --parallel ${_flatc_jobs}
        WORKING_DIRECTORY "${_root_host_flatc_dir}"
        RESULT_VARIABLE HOSTBUILD_RC
        ERROR_VARIABLE HOSTBUILD_ERR)
      if(NOT HOSTBUILD_RC EQUAL 0)
        message(FATAL_ERROR "Failed to build host flatc:\n${HOSTBUILD_ERR}")
      endif()
      if(EXISTS "${_root_host_flatc_dir}/_deps/flatbuffers-build/Release/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
        set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build/Release" CACHE PATH "Host tools directory")
      else()
        set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build" CACHE PATH "Host tools directory")
      endif()
    endif()
  endif()
endif()

# Add TFLite as a subdirectory
if(EXISTS "${TFLITE_SOURCE_DIR}/CMakeLists.txt")
  add_subdirectory("${TFLITE_SOURCE_DIR}" "${TFLITE_BUILD_DIR}" EXCLUDE_FROM_ALL)
  if(WIN32 AND TARGET tensorflow-lite)
    # simple_memory_arena_debug_dump.cc is the strong override of the
    # TFLITE_ATTRIBUTE_WEAK DumpArenaInfo() in simple_memory_arena.cc. PE/COFF
    # has no weak symbols, so core/macros.h defines TFLITE_ATTRIBUTE_WEAK away
    # on Windows and both definitions become strong: linking the two objects
    # together fails with LNK2005 when LiteRt.dll whole-archives
    # tensorflow-lite. TFLite's own CMakeLists.txt only filters this file out
    # when BUILD_SHARED_LIBS is ON.
    get_target_property(_litert_tflite_srcs tensorflow-lite SOURCES)
    if(_litert_tflite_srcs)
      list(FILTER _litert_tflite_srcs EXCLUDE REGEX ".*simple_memory_arena_debug_dump\\.cc$")
      set_target_properties(tensorflow-lite PROPERTIES SOURCES "${_litert_tflite_srcs}")
    endif()
  endif()
else()
  message(FATAL_ERROR "TFLite source directory NOT found at ${TFLITE_SOURCE_DIR}.
    Please ensure TFLite source is available or set TFLITE_SOURCE_DIR.")
endif()


# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TFLITE_SOURCE_DIR}
)

list(APPEND CMAKE_MODULE_PATH "${TFLITE_SOURCE_DIR}/tools/cmake/modules")

if(NOT EXISTS "${CMAKE_BINARY_DIR}/opencl_headers")
  find_package(opencl_headers REQUIRED)
endif()
include_directories(${CMAKE_BINARY_DIR}/opencl_headers)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/vulkan_headers")
  find_package(vulkan_headers REQUIRED)
endif()
include_directories(AFTER "${CMAKE_BINARY_DIR}/vulkan_headers/include")

if(NOT EXISTS "${CMAKE_BINARY_DIR}/opengl_headers")
  find_package(opengl_headers REQUIRED)
endif()
include_directories(AFTER "${CMAKE_BINARY_DIR}/opengl_headers/api")

if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/egl_headers")
    find_package(egl_headers REQUIRED)
  endif()
  include_directories(AFTER "${CMAKE_BINARY_DIR}/egl_headers/api")
endif()

# Add FP16 headers include directory
if(EXISTS "${CMAKE_BINARY_DIR}/FP16-source/include")
  include_directories(${CMAKE_BINARY_DIR}/FP16-source/include)
elseif(NOT EXISTS "${CMAKE_BINARY_DIR}/fp16_headers")
  find_package(fp16_headers REQUIRED)
else()
  include_directories(${CMAKE_BINARY_DIR}/fp16_headers/include)
endif()

# Add SafeInt headers include directory
include(OverridableFetchContent)
OverridableFetchContent_Declare(
  safeint
  GIT_REPOSITORY https://github.com/dcleblanc/SafeInt
  GIT_TAG 72f0745e26091af6d1186fd91426377f16688ba7
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/safeint"
)
OverridableFetchContent_GetProperties(safeint)
if(NOT safeint_POPULATED)
  OverridableFetchContent_Populate(safeint)
endif()
include_directories(${safeint_SOURCE_DIR})

# Add ML Drift headers include directory
OverridableFetchContent_Declare(
  ml_drift
  GIT_REPOSITORY https://github.com/google-ai-edge/ml-drift.git
  GIT_TAG 1b18771f44c0be3d972847b31a80f6dc3d6964b5
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/ml_drift"
)
OverridableFetchContent_GetProperties(ml_drift)
if(NOT ml_drift_POPULATED)
  OverridableFetchContent_Populate(ml_drift)
endif()
include_directories(${ml_drift_SOURCE_DIR})

# Find FlatBuffers
#
# TFLite is added via add_subdirectory(... "${TFLITE_BUILD_DIR}"), but its
# FetchContent of FlatBuffers materializes under ${CMAKE_BINARY_DIR} (the
# top-level binary dir, which does not change inside add_subdirectory), NOT
# under ${TFLITE_BUILD_DIR}. Point at ${CMAKE_BINARY_DIR} so the headers and
# library resolve for targets that reference FlatBuffers directly (e.g.
# qnn_flexbuffer_helpers) rather than transitively through tensorflow-lite.
set(LITERT_FLATBUFFERS_DIR "${CMAKE_BINARY_DIR}/flatbuffers")
set(LITERT_FLATBUFFERS_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/flatbuffers-build")
find_package(flatbuffers QUIET
    PATHS
      ${LITERT_FLATBUFFERS_BUILD_DIR}
      ${LITERT_FLATBUFFERS_DIR}/CMake
      ${CMAKE_BINARY_DIR}/flatbuffers-flatc/lib/cmake/flatbuffers
    NO_DEFAULT_PATH
)

if(NOT TARGET flatbuffers::flatbuffers)
  if(TARGET flatbuffers)
    add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
  else()
    add_library(flatbuffers::flatbuffers STATIC IMPORTED)
    set_target_properties(flatbuffers::flatbuffers PROPERTIES
      IMPORTED_LOCATION "${LITERT_FLATBUFFERS_BUILD_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}flatbuffers${CMAKE_STATIC_LIBRARY_SUFFIX}"
      INTERFACE_INCLUDE_DIRECTORIES "${LITERT_FLATBUFFERS_DIR}/include"
    )
  endif()
endif()

# Locate flatc for generating ML Drift FlatBuffer headers
set(LITERT_FLATC_EXECUTABLE "")
set(LITERT_FLATC_TARGET "")
if(CMAKE_CROSSCOMPILING AND TFLITE_HOST_TOOLS_DIR)
  foreach(_litert_flatc_candidate
          "${TFLITE_HOST_TOOLS_DIR}/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}"
          "${TFLITE_HOST_TOOLS_DIR}/Release/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}"
          "${TFLITE_HOST_TOOLS_DIR}/bin/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}"
          "${TFLITE_HOST_TOOLS_DIR}/flatbuffers-flatc/bin/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
    if(EXISTS "${_litert_flatc_candidate}")
      set(LITERT_FLATC_EXECUTABLE "${_litert_flatc_candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT LITERT_FLATC_EXECUTABLE)
  if(TARGET flatbuffers::flatc)
    set(LITERT_FLATC_EXECUTABLE "$<TARGET_FILE:flatbuffers::flatc>")
    set(LITERT_FLATC_TARGET flatbuffers::flatc)
  elseif(TARGET flatc)
    set(LITERT_FLATC_EXECUTABLE "$<TARGET_FILE:flatc>")
    set(LITERT_FLATC_TARGET flatc)
  elseif(EXISTS "${CMAKE_BINARY_DIR}/flatbuffers-flatc/bin/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
    set(LITERT_FLATC_EXECUTABLE "${CMAKE_BINARY_DIR}/flatbuffers-flatc/bin/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
  elseif(EXISTS "${LITERT_FLATBUFFERS_BUILD_DIR}/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
    set(LITERT_FLATC_EXECUTABLE "${LITERT_FLATBUFFERS_BUILD_DIR}/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
  else()
    unset(LITERT_FLATC_EXECUTABLE)
    find_program(LITERT_FLATC_EXECUTABLE NAMES flatc)
  endif()
endif()

set(MLDRIFT_SERIALIZATION_BASE_FBS "${ml_drift_SOURCE_DIR}/ml_drift/common/task/serialization_base.fbs")
set(MLDRIFT_SERIALIZATION_BASE_H "${ml_drift_SOURCE_DIR}/ml_drift/common/task/serialization_base_generated.h")
set(MLDRIFT_CL_PROGRAM_CACHE_FBS "${ml_drift_SOURCE_DIR}/ml_drift/cl/compiled_program_cache.fbs")
set(MLDRIFT_CL_PROGRAM_CACHE_H "${ml_drift_SOURCE_DIR}/ml_drift/cl/compiled_program_cache_generated.h")
add_custom_command(
  OUTPUT "${MLDRIFT_SERIALIZATION_BASE_H}" "${MLDRIFT_CL_PROGRAM_CACHE_H}"
  COMMAND ${LITERT_FLATC_EXECUTABLE} --cpp --scoped-enums
          -I "${ml_drift_SOURCE_DIR}"
          -o "${ml_drift_SOURCE_DIR}/ml_drift/common/task"
          "${MLDRIFT_SERIALIZATION_BASE_FBS}"
  COMMAND ${LITERT_FLATC_EXECUTABLE} --cpp --scoped-enums
          -I "${ml_drift_SOURCE_DIR}"
          -o "${ml_drift_SOURCE_DIR}/ml_drift/cl"
          "${MLDRIFT_CL_PROGRAM_CACHE_FBS}"
  DEPENDS "${MLDRIFT_SERIALIZATION_BASE_FBS}" "${MLDRIFT_CL_PROGRAM_CACHE_FBS}" ${LITERT_FLATC_TARGET}
  COMMENT "Generating ML Drift FlatBuffer headers"
)
add_custom_target(mldrift_serialization_gen DEPENDS
  "${MLDRIFT_SERIALIZATION_BASE_H}"
  "${MLDRIFT_CL_PROGRAM_CACHE_H}"
)

# Find Abseil
find_package(absl REQUIRED)

# Platform detection
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LITERT_PLATFORM_MACOS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LITERT_PLATFORM_LINUX ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LITERT_PLATFORM_WINDOWS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(LITERT_PLATFORM_ANDROID ON)
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Add subdirectories
add_subdirectory(c)
add_subdirectory(cc)
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(compiler)
# vendors/ downloads the vendor SDKs (e.g. the multi-GB Qualcomm QAIRT SDK) at
# configure time, so only add it when NPU support is enabled.
if(LITERT_ENABLE_NPU)
  add_subdirectory(vendors)
else()
  message(STATUS "Skipping NPU vendor dispatch libraries: LITERT_ENABLE_NPU is OFF")
endif()
add_subdirectory(tools)
add_subdirectory(experimental/custom_ops/gated_delta_net)


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tensor/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../tensor"
                   "${CMAKE_CURRENT_BINARY_DIR}/tensor")
endif()

if(LITERT_BUILD_SUPPORT)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../support"
                   "${CMAKE_CURRENT_BINARY_DIR}/support")
endif()

