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
project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)


# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress deprecation warnings (matching Bazel build behavior in .bazelrc)
# The codebase has deprecated Error constructors that are still in use
add_compile_options(
  -Wno-deprecated-declarations
  -Wno-return-type
  -fpermissive
)

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
option(LITERT_BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(LITERT_BUILD_TOOLS "Build LiteRT tools" ON)
option(LITERT_BUILD_TESTS "Build LiteRT tests" OFF)
option(LITERT_AUTO_BUILD_TFLITE "Automatically build TFLite if not found" ON)
option(LITERT_ENABLE_GPU "Enable GPU acceleration support" ON)
option(LITERT_ENABLE_NPU "Enable NPU acceleration support" ON)
option(LITERT_DISABLE_KLEIDIAI "Disable KleidiAI delegate when bundling host TFLite (default ON for Linux/macOS x86_64 hosts)" ${_litert_disable_kleidiai_default})

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

set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "Enable XNNPACK in TFLite")
set(TFLITE_ENABLE_GPU ON CACHE BOOL "Enable GPU in TFLite")
if(LITERT_DISABLE_KLEIDIAI)
  set(XNNPACK_ENABLE_KLEIDIAI OFF CACHE BOOL "Disable KleidiAI in XNNPACK")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Build or reuse host flatc for cross compilation.
  if(NOT TFLITE_HOST_TOOLS_DIR)
    set(_root_host_flatc_dir "${CMAKE_CURRENT_SOURCE_DIR}/../host_flatc_build")
    if(EXISTS "${_root_host_flatc_dir}/_deps/flatbuffers-build/flatc")
      set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build")
      message(STATUS "Reusing host flatc from ${TFLITE_HOST_TOOLS_DIR}")
    else()
      file(MAKE_DIRECTORY "${_root_host_flatc_dir}")
      file(WRITE "${_root_host_flatc_dir}/CMakeLists.txt" "cmake_minimum_required(VERSION 3.16)\nproject(HostFlatc LANGUAGES C CXX)\ninclude(FetchContent)\nFetchContent_Declare(flatbuffers GIT_REPOSITORY https://github.com/google/flatbuffers.git GIT_TAG v25.9.23)\nset(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\nFetchContent_MakeAvailable(flatbuffers)\n")
      set(_litert_host_flatc_cmake_args -S . -B . -DCMAKE_BUILD_TYPE=Release)
      set(_litert_host_c_compiler "${LITERT_HOST_C_COMPILER}")
      if(NOT _litert_host_c_compiler)
        if(NOT CMAKE_CROSSCOMPILING AND CMAKE_C_COMPILER)
          set(_litert_host_c_compiler "${CMAKE_C_COMPILER}")
        elseif(EXISTS "/usr/bin/clang")
          set(_litert_host_c_compiler "/usr/bin/clang")
        endif()
      endif()
      if(_litert_host_c_compiler)
        list(APPEND _litert_host_flatc_cmake_args -DCMAKE_C_COMPILER=${_litert_host_c_compiler})
      endif()
      set(_litert_host_cxx_compiler "${LITERT_HOST_CXX_COMPILER}")
      if(NOT _litert_host_cxx_compiler)
        if(NOT CMAKE_CROSSCOMPILING AND CMAKE_CXX_COMPILER)
          set(_litert_host_cxx_compiler "${CMAKE_CXX_COMPILER}")
        elseif(EXISTS "/usr/bin/clang++")
          set(_litert_host_cxx_compiler "/usr/bin/clang++")
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
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --target flatc -- -j${CMAKE_BUILD_PARALLEL_LEVEL}
        WORKING_DIRECTORY "${_root_host_flatc_dir}"
        RESULT_VARIABLE HOSTBUILD_RC
        ERROR_VARIABLE HOSTBUILD_ERR)
      if(NOT HOSTBUILD_RC EQUAL 0)
        message(FATAL_ERROR "Failed to build host flatc:\n${HOSTBUILD_ERR}")
      endif()
      set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build" CACHE PATH "Host tools directory")
    endif()
  endif()
endif()

# Add TFLite as a subdirectory
if(EXISTS "${TFLITE_SOURCE_DIR}/CMakeLists.txt")
  add_subdirectory("${TFLITE_SOURCE_DIR}" "${TFLITE_BUILD_DIR}" EXCLUDE_FROM_ALL)
else()
  message(FATAL_ERROR "TFLite source directory NOT found at ${TFLITE_SOURCE_DIR}.
    Please ensure TFLite source is available or set TFLITE_SOURCE_DIR.")
endif()


# Get TensorFlow source directory from TFLite build
if(EXISTS "${TFLITE_BUILD_DIR}")
    include("${TFLITE_BUILD_DIR}/tensorflow_src_dir.cmake" OPTIONAL)
endif()

# Set default TensorFlow source directory if not found
if(NOT TENSORFLOW_SOURCE_DIR)
    if(EXISTS "${TFLITE_BUILD_DIR}/tensorflow-src")
        set(TENSORFLOW_SOURCE_DIR "${TFLITE_BUILD_DIR}/tensorflow-src")
    else()
        message(FATAL_ERROR "TensorFlow source directory not found. Please build TFLite first or set TENSORFLOW_SOURCE_DIR")
    endif()
endif()

message(STATUS "Using TensorFlow source directory: ${TENSORFLOW_SOURCE_DIR}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TFLITE_SOURCE_DIR}
    ${TENSORFLOW_SOURCE_DIR}
)

list(APPEND CMAKE_MODULE_PATH "${TFLITE_SOURCE_DIR}/tools/cmake/modules")

if(EXISTS "${CMAKE_BINARY_DIR}/opencl_headers")
    include_directories(${CMAKE_BINARY_DIR}/opencl_headers)
endif()

# Add FP16 headers include directory
if(EXISTS "${CMAKE_BINARY_DIR}/FP16-source/include")
    include_directories(${CMAKE_BINARY_DIR}/FP16-source/include)
endif()

# Find FlatBuffers
find_package(flatbuffers QUIET
    PATHS
      ${TFLITE_BUILD_DIR}/_deps/flatbuffers-build
      ${TFLITE_BUILD_DIR}/flatbuffers/CMake
      ${TFLITE_BUILD_DIR}/flatbuffers-flatc/lib/cmake/flatbuffers
    NO_DEFAULT_PATH
)

if(NOT TARGET flatbuffers::flatbuffers)
  add_library(flatbuffers::flatbuffers STATIC IMPORTED)
  set_target_properties(flatbuffers::flatbuffers PROPERTIES
    IMPORTED_LOCATION "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/libflatbuffers.a"
    INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_BUILD_DIR}/flatbuffers/include"
  )
endif()

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

# Add subdirectories
add_subdirectory(c)
add_subdirectory(cc)
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(compiler)
add_subdirectory(vendors)
if(LITERT_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
