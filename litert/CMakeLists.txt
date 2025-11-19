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

function(_litert_overlay_converter_sources TF_SRC_ROOT)
  set(_overlay_root "${CMAKE_CURRENT_SOURCE_DIR}/../tflite/converter")
  if(NOT EXISTS "${TF_SRC_ROOT}")
    message(WARNING "LiteRT: TFLite source root '${TF_SRC_ROOT}' not found; converter overlay skipped")
    return()
  endif()

  set(_overlay_srcs
    "${_overlay_root}/allocation.h"
    "${_overlay_root}/mmap_allocation.cc"
    "${_overlay_root}/mmap_allocation_disabled.cc"
    "${_overlay_root}/core/model_builder_base.cc"
    "${_overlay_root}/core/model_builder_base.h"
  )

  set(_overlay_dests
    "tensorflow/compiler/mlir/lite/allocation.h"
    "tensorflow/compiler/mlir/lite/mmap_allocation.cc"
    "tensorflow/compiler/mlir/lite/mmap_allocation_disabled.cc"
    "tensorflow/compiler/mlir/lite/core/model_builder_base.cc"
    "tensorflow/compiler/mlir/lite/core/model_builder_base.h"
  )

  list(LENGTH _overlay_srcs _count)
  math(EXPR _last "${_count} - 1")
  foreach(_idx RANGE ${_last})
    list(GET _overlay_srcs ${_idx} _src)
    list(GET _overlay_dests ${_idx} _rel_dst)
    if(NOT EXISTS "${_src}")
      message(WARNING "LiteRT: converter overlay source '${_src}' missing")
      continue()
    endif()
    set(_dst "${TF_SRC_ROOT}/${_rel_dst}")
    get_filename_component(_dst_dir "${_dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_dir}")
    file(READ "${_src}" _content)
    string(REPLACE "\"tflite/converter/" "\"tensorflow/compiler/mlir/lite/" _content "${_content}")
    string(REPLACE "<tflite/converter/" "<tensorflow/compiler/mlir/lite/" _content "${_content}")
    file(WRITE "${_dst}" "${_content}")
  endforeach()
endfunction()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(LITERT_BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(LITERT_BUILD_TOOLS "Build LiteRT tools" ON)
option(LITERT_BUILD_TESTS "Build LiteRT tests" OFF)
option(LITERT_AUTO_BUILD_TFLITE "Automatically build TFLite if not found" ON)
option(LITERT_ENABLE_GPU "Enable GPU acceleration support" ON)
option(LITERT_ENABLE_NPU "Enable NPU acceleration support" ON)

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

# Find TFLite package - we depend on it being built first
set(TFLITE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/tflite_build" CACHE PATH "Path to TFLite build directory")
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite" CACHE PATH "Path to TFLite source directory")

# Check if TFLite has been built
if(NOT EXISTS "${TFLITE_BUILD_DIR}/libtensorflow-lite.a")
    if(LITERT_AUTO_BUILD_TFLITE)
        message(STATUS "TFLite not found at ${TFLITE_BUILD_DIR}, building it automatically...")
        message(STATUS "This may take 5-10 minutes on first build...")
        message(STATUS "To skip automatic build, set -DLITERT_AUTO_BUILD_TFLITE=OFF")

        # Create TFLite build directory
        file(MAKE_DIRECTORY "${TFLITE_BUILD_DIR}")

        # Configure TFLite
        message(STATUS "Configuring TFLite...")
        set(_tflite_extra_args -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
        set(_litert_disable_kleidiai FALSE)
        if(CMAKE_SYSTEM_PROCESSOR)
          string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _litert_system_proc_lower)
        else()
          set(_litert_system_proc_lower "")
        endif()
        if(_litert_system_proc_lower MATCHES "^(x86_64|amd64|i[3-6]86)$")
          if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(_litert_disable_kleidiai TRUE)
          endif()
        endif()
        if(_litert_disable_kleidiai)
          list(APPEND _tflite_extra_args -DXNNPACK_ENABLE_KLEIDIAI=OFF)
          message(STATUS "Disabling KleidiAI for host ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        if(CMAKE_SYSTEM_NAME STREQUAL "Android")
          # Propagate Android toolchain settings to the inner TFLite configure
          if(CMAKE_TOOLCHAIN_FILE)
            list(APPEND _tflite_extra_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
          endif()
          if(DEFINED ANDROID_ABI)
            list(APPEND _tflite_extra_args -DANDROID_ABI=${ANDROID_ABI})
          endif()
          if(DEFINED ANDROID_PLATFORM)
            list(APPEND _tflite_extra_args -DANDROID_PLATFORM=${ANDROID_PLATFORM})
          endif()
          list(APPEND _tflite_extra_args -DCMAKE_SYSTEM_NAME=Android)
          list(APPEND _tflite_extra_args -DTFLITE_ENABLE_GPU=ON)

          # Build or reuse host flatc for cross compilation.
          if(NOT TFLITE_HOST_TOOLS_DIR)
            set(_root_host_flatc_dir "${CMAKE_CURRENT_SOURCE_DIR}/../host_flatc_build")
            if(EXISTS "${_root_host_flatc_dir}/_deps/flatbuffers-build/flatc")
              set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build")
              message(STATUS "Reusing host flatc from ${TFLITE_HOST_TOOLS_DIR}")
            else()
              file(MAKE_DIRECTORY "${_root_host_flatc_dir}")
              file(WRITE "${_root_host_flatc_dir}/CMakeLists.txt" "cmake_minimum_required(VERSION 3.16)\nproject(HostFlatc LANGUAGES C CXX)\ninclude(FetchContent)\nFetchContent_Declare(flatbuffers GIT_REPOSITORY https://github.com/google/flatbuffers.git GIT_TAG v25.9.23)\nset(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\nFetchContent_MakeAvailable(flatbuffers)\n")
              execute_process(
                COMMAND ${CMAKE_COMMAND} -S . -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++
                WORKING_DIRECTORY "${_root_host_flatc_dir}"
                RESULT_VARIABLE HOSTCFG_RC
                ERROR_VARIABLE HOSTCFG_ERR)
              if(NOT HOSTCFG_RC EQUAL 0)
                message(FATAL_ERROR "Failed to configure host flatbuffers:\n${HOSTCFG_ERR}")
              endif()
              execute_process(
                COMMAND ${CMAKE_COMMAND} --build . --target flatc -- -j
                WORKING_DIRECTORY "${_root_host_flatc_dir}"
                RESULT_VARIABLE HOSTBUILD_RC
                ERROR_VARIABLE HOSTBUILD_ERR)
              if(NOT HOSTBUILD_RC EQUAL 0)
                message(FATAL_ERROR "Failed to build host flatc:\n${HOSTBUILD_ERR}")
              endif()
              set(TFLITE_HOST_TOOLS_DIR "${_root_host_flatc_dir}/_deps/flatbuffers-build")
              message(STATUS "Host flatc prepared at ${TFLITE_HOST_TOOLS_DIR}")
            endif()
          endif()

          list(APPEND _tflite_extra_args -DTFLITE_HOST_TOOLS_DIR=${TFLITE_HOST_TOOLS_DIR})
        endif()
        execute_process(
            COMMAND ${CMAKE_COMMAND} "${TFLITE_SOURCE_DIR}"
                -DTFLITE_ENABLE_XNNPACK=ON
                -DTFLITE_ENABLE_GPU=ON
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                ${_tflite_extra_args}
            WORKING_DIRECTORY "${TFLITE_BUILD_DIR}"
            RESULT_VARIABLE TFLITE_CONFIGURE_RESULT
            OUTPUT_VARIABLE TFLITE_CONFIGURE_OUTPUT
            ERROR_VARIABLE TFLITE_CONFIGURE_ERROR
        )

        if(NOT TFLITE_CONFIGURE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to configure TFLite:\n${TFLITE_CONFIGURE_ERROR}")
        endif()

        # In this build. TFLite requires FlatBuffers 25.9.23.
        set(FLATBUFFERS_REQUIRED_MAJOR 25)
        set(FLATBUFFERS_REQUIRED_MINOR 9)
        set(FLATBUFFERS_REQUIRED_REVISION 23)
        set(TF_SRC_DIR "${TFLITE_BUILD_DIR}/tensorflow-src")
        if(EXISTS "${TF_SRC_DIR}")
            _litert_overlay_converter_sources("${TF_SRC_DIR}")
            file(GLOB_RECURSE TF_FB_GENERATED_HDRS
                 "${TF_SRC_DIR}/tensorflow/*.h")
            foreach(HDR IN LISTS TF_FB_GENERATED_HDRS)
                file(READ "${HDR}" _cnt)
                if(_cnt MATCHES "static_assert\\(FLATBUFFERS_VERSION_MAJOR")
                    set(_new "${_cnt}")
                    string(REGEX REPLACE "FLATBUFFERS_VERSION_MAJOR[ ]*==[ ]*[0-9]+"
                           "FLATBUFFERS_VERSION_MAJOR == ${FLATBUFFERS_REQUIRED_MAJOR}"
                           _new "${_new}")
                    string(REGEX REPLACE "FLATBUFFERS_VERSION_MINOR[ ]*==[ ]*[0-9]+"
                           "FLATBUFFERS_VERSION_MINOR == ${FLATBUFFERS_REQUIRED_MINOR}"
                           _new "${_new}")
                    string(REGEX REPLACE "FLATBUFFERS_VERSION_REVISION[ ]*==[ ]*[0-9]+"
                           "FLATBUFFERS_VERSION_REVISION == ${FLATBUFFERS_REQUIRED_REVISION}"
                           _new "${_new}")
                    if(NOT _new STREQUAL _cnt)
                        file(WRITE "${HDR}" "${_new}")
                    endif()
                endif()
            endforeach()
        endif()

        # Build TFLite
        message(STATUS "Building TFLite (this will take several minutes)...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build . --target tensorflow-lite -- -j
            WORKING_DIRECTORY "${TFLITE_BUILD_DIR}"
            RESULT_VARIABLE TFLITE_BUILD_RESULT
            OUTPUT_QUIET
            ERROR_VARIABLE TFLITE_BUILD_ERROR
        )

        if(NOT TFLITE_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build TFLite:\n${TFLITE_BUILD_ERROR}")
        endif()

        message(STATUS "TFLite build completed successfully")
    else()
        message(FATAL_ERROR "TFLite not found at ${TFLITE_BUILD_DIR}. Please build TFLite first or set LITERT_AUTO_BUILD_TFLITE=ON")
    endif()
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
        message(FATAL_ERROR "TensorFlow source directory not found. Please build TFLite first or set
ENSORFLOW_SOURCE_DIR")
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

if(EXISTS "${TFLITE_BUILD_DIR}/opencl_headers")
    include_directories(${TFLITE_BUILD_DIR}/opencl_headers)
endif()

# Add FP16 headers include directory
if(EXISTS "${TFLITE_BUILD_DIR}/FP16-source/include")
    include_directories(${TFLITE_BUILD_DIR}/FP16-source/include)
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

if(NOT TARGET tensorflow-lite)
  if(EXISTS "${TFLITE_BUILD_DIR}/libtensorflow-lite.a")
    add_library(tensorflow-lite STATIC IMPORTED)
    set(_litert_tflite_link_libs
      "${TFLITE_BUILD_DIR}/pthreadpool/libpthreadpool.a"
      "${TFLITE_BUILD_DIR}/_deps/farmhash-build/libfarmhash.a"
      "${TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_fftsg2d.a"
      "${TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_fftsg.a"
      "${TFLITE_BUILD_DIR}/_deps/cpuinfo-build/libcpuinfo.a"
      "${TFLITE_BUILD_DIR}/libxnnpack-delegate.a"
      "${TFLITE_BUILD_DIR}/_deps/xnnpack-build/libXNNPACK.a"
      "${TFLITE_BUILD_DIR}/_deps/xnnpack-build/libxnnpack-microkernels-prod.a"
    )
    set(_kleidiai_lib "${TFLITE_BUILD_DIR}/kleidiai/libkleidiai.a")
    if(EXISTS "${_kleidiai_lib}")
      list(APPEND _litert_tflite_link_libs "${_kleidiai_lib}")
    else()
      message(STATUS "LiteRT: KleidiAI static library not found at ${_kleidiai_lib}; continuing without it.")
    endif()
    list(APPEND _litert_tflite_link_libs
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_allocator.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_apply_multiplier.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_block_map.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_blocking_counter.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_context.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_context_get_ctx.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_cpuinfo.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_ctx.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_denormal.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_frontend.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_have_built_path_for_avx.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_have_built_path_for_avx2_fma.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_have_built_path_for_avx512.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_kernel_arm.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_kernel_avx.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_kernel_avx2_fma.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_kernel_avx512.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_pack_arm.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_pack_avx.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_pack_avx2_fma.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_pack_avx512.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_prepare_packed_matrices.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_prepacked_cache.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_system_aligned_alloc.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_thread_pool.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/profiler/libruy_profiler_instrumentation.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_trmul.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_tune.a"
      "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy_wait.a"
      "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/libflatbuffers.a"
    )
    set_target_properties(tensorflow-lite PROPERTIES
      IMPORTED_LOCATION "${TFLITE_BUILD_DIR}/libtensorflow-lite.a"
      INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_SOURCE_DIR};${TFLITE_BUILD_DIR}"
      INTERFACE_LINK_LIBRARIES "${_litert_tflite_link_libs}"
    )
  endif()
endif()

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
