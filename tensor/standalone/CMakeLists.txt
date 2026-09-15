# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standalone tensor/XNNPACK build: cmake -S tensor/standalone -B <build-dir> -G Ninja.
# For Android, select the NDK toolchain at configure time. Build the same test
# executables, then use the Android test launcher rather than executing CTest on
# the build host. Keep this source/test inventory synchronized with BUILD files.
cmake_minimum_required(VERSION 3.28)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(LiteRtNativeTensor LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)

include(FetchContent)
enable_testing()

get_filename_component(LITERT_TENSOR_SOURCE_DIR ".." ABSOLUTE
                       BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(LITERT_SOURCE_ROOT "../.." ABSOLUTE
                       BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Keep the first consolidated build on the revision used by the validated
# native runner. Dependencies are downloaded; no other source checkout or old
# build archive is required. FETCHCONTENT_SOURCE_DIR_LITERT_NATIVE_XNNPACK can
# explicitly select a source tree for a later kernel experiment.
FetchContent_Declare(litert_native_xnnpack
  URL https://codeload.github.com/google/XNNPACK/tar.gz/bf3ee43b63070284f85a4a882f298b6ca5273f01
  URL_HASH SHA256=a8fb7cb8684df1f3a18230f4f66cfd215c5340c9d1e5f597f76c30d9d027721b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL)

# Match the pinned XNNPACK MODULE.bazel, including its Google Test override. Standard
# FETCHCONTENT_SOURCE_DIR_GOOGLETEST / _ABSEIL overrides remain available for
# offline builds; no existing Bazel checkout or cache is required.
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/8eff9e336692fc95961e096564f1044c600b881d.zip
  URL_HASH SHA256=6e8db0498a4fdcaa0b03bc4d554d7af8d0a84f49e8d8ba3e5509a6419fa066a5
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  abseil
  URL https://github.com/abseil/abseil-cpp/releases/download/20260107.1/abseil-cpp-20260107.1.tar.gz
  URL_HASH SHA256=4314e2a7cbac89cac25a2f2322870f343d81579756ceff7f431803c2c9090195
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TEST_HELPERS ON CACHE BOOL "" FORCE)
set(ABSL_USE_EXTERNAL_GOOGLETEST ON CACHE BOOL "" FORCE)
set(ABSL_FIND_GOOGLETEST OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(abseil)

# pthreadpool's CMake fallback tracks FXdiv's master branch. Supply the same
# fixed archive as XNNPACK's MODULE.bazel instead; honor a source override.
if(NOT DEFINED FXDIV_SOURCE_DIR)
  FetchContent_Declare(
    fxdiv
    URL https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip
    URL_HASH SHA256=ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(FXDIV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(fxdiv)
  set(FXDIV_SOURCE_DIR "${fxdiv_SOURCE_DIR}" CACHE PATH "FXdiv source directory")
endif()

# Reuse XNNPACK's supported CMake cross-compilation path and its pinned
# dependency downloads. Runtime tests need production microkernels only.
set(XNNPACK_LIBRARY_TYPE static CACHE STRING "" FORCE)
set(XNNPACK_BUILD_LIBRARY ON CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_ALL_MICROKERNELS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_YNNPACK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(litert_native_xnnpack)
set(LITERT_TENSOR_XNNPACK_ROOT "${litert_native_xnnpack_SOURCE_DIR}")

# Header dependencies shared by the public API and graph implementation.
add_library(litert_tensor_headers INTERFACE)
target_compile_definitions(litert_tensor_headers INTERFACE LITERT_TENSOR_STANDALONE=1)
target_include_directories(litert_tensor_headers INTERFACE
  "${LITERT_SOURCE_ROOT}" "${LITERT_TENSOR_XNNPACK_ROOT}"
  "${LITERT_TENSOR_XNNPACK_ROOT}/src")
target_compile_features(litert_tensor_headers INTERFACE cxx_std_17)
target_compile_options(litert_tensor_headers INTERFACE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>")
target_link_libraries(litert_tensor_headers INTERFACE
  absl::absl_check
  absl::absl_log
  absl::algorithm_container
  absl::base
  absl::flat_hash_map
  absl::flat_hash_set
  absl::inlined_vector
  absl::no_destructor
  absl::optional
  absl::random_random
  absl::span
  absl::status
  absl::statusor
  absl::str_format
  absl::strings
  absl::string_view)

# Compile internal/graph.cc independently in the ordinary and debugger-enabled cores.
# All other core objects are shared; each executable links exactly one graph
# implementation, so enabling the debugger does not change ordinary tests.
add_library(litert_tensor_core_objects OBJECT
  "${LITERT_TENSOR_SOURCE_DIR}/buffer.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/tensor.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/internal/shape.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/internal/graph_probe.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/internal/graph_traversal.cc")
target_link_libraries(litert_tensor_core_objects PUBLIC litert_tensor_headers)

add_library(litert_tensor_core STATIC
  "${LITERT_TENSOR_SOURCE_DIR}/internal/graph.cc" $<TARGET_OBJECTS:litert_tensor_core_objects>)
target_link_libraries(litert_tensor_core PUBLIC litert_tensor_headers)

add_library(litert_tensor_core_debugger STATIC
  "${LITERT_TENSOR_SOURCE_DIR}/internal/graph.cc" $<TARGET_OBJECTS:litert_tensor_core_objects>)
target_compile_definitions(litert_tensor_core_debugger PRIVATE
                           LITERT_OP_DEBUGGER_ENABLED=1)
target_link_libraries(litert_tensor_core_debugger PUBLIC litert_tensor_headers)

add_library(litert_tensor_common_backend STATIC
  "${LITERT_TENSOR_SOURCE_DIR}/backends/common_nnpack/conversion.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/backends/common_nnpack/graph.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/backends/common_nnpack/utils.cc")
target_link_libraries(litert_tensor_common_backend PUBLIC litert_tensor_core)

add_library(litert_tensor_xnnpack_backend STATIC
  "${LITERT_TENSOR_SOURCE_DIR}/backends/xnnpack/arithmetic.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/backends/xnnpack/conversion.cc")
target_link_libraries(litert_tensor_xnnpack_backend PUBLIC
                      litert_tensor_common_backend XNNPACK)

add_library(litert_tensor_common_runner STATIC "${LITERT_TENSOR_SOURCE_DIR}/runners/common_nnpack/runner.cc")
target_link_libraries(litert_tensor_common_runner PUBLIC
                      litert_tensor_common_backend)

add_library(litert_tensor_xnnpack_runner STATIC "${LITERT_TENSOR_SOURCE_DIR}/runners/xnnpack/runner.cc")
target_link_libraries(litert_tensor_xnnpack_runner PUBLIC
  litert_tensor_common_runner litert_tensor_xnnpack_backend pthreadpool)

# CTest and the Android launcher consume the same explicit test inventory.
# One CTest entry corresponds to one Bazel cc_test binary, preserving Google
# Test's own case names, skips, filtering, repeats, and XML output behavior.
function(litert_tensor_add_test name source library)
  cmake_parse_arguments(TEST "" "FRAMEWORK" "LABELS;LIBRARIES" ${ARGN})
  if(NOT TEST_FRAMEWORK)
    set(TEST_FRAMEWORK gtest)
  endif()
  add_executable(${name} "${source}")
  target_link_libraries(${name} PRIVATE
    ${library} GTest::gtest_main GTest::gmock absl::status_matchers
    ${TEST_LIBRARIES})
  # A generator expression prevents multi-config generators adding a config
  # suffix; the manifest consequently has the same paths for every config.
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin/$<0:>")
  add_test(NAME ${name} COMMAND ${name})
  set_property(TEST ${name} PROPERTY LABELS tensor ${TEST_LABELS})

  set(labels "\"tensor\"")
  foreach(label IN LISTS TEST_LABELS)
    string(APPEND labels ", \"${label}\"")
  endforeach()
  set_property(GLOBAL APPEND PROPERTY LITERT_TENSOR_TEST_MANIFEST_ENTRIES
    "    {\"name\": \"${name}\", \"binary\": \"$<TARGET_FILE:${name}>\", \"labels\": [${labels}], \"framework\": \"${TEST_FRAMEWORK}\"}")
endfunction()

litert_tensor_add_test(litert_tensor_buffer_test "${LITERT_TENSOR_SOURCE_DIR}/buffer_test.cc"
  litert_tensor_core LABELS core)
litert_tensor_add_test(litert_tensor_datatypes_test "${LITERT_TENSOR_SOURCE_DIR}/datatypes_test.cc"
  litert_tensor_core LABELS core)
litert_tensor_add_test(litert_tensor_tensor_test "${LITERT_TENSOR_SOURCE_DIR}/tensor_test.cc"
  litert_tensor_core LABELS core)
litert_tensor_add_test(litert_tensor_arithmetic_test "${LITERT_TENSOR_SOURCE_DIR}/arithmetic_test.cc"
  litert_tensor_core LABELS core)

litert_tensor_add_test(litert_tensor_internal_graph_test "${LITERT_TENSOR_SOURCE_DIR}/internal/graph_test.cc"
  litert_tensor_core LABELS internal)
litert_tensor_add_test(litert_tensor_internal_graph_traversal_test
  "${LITERT_TENSOR_SOURCE_DIR}/internal/graph_traversal_test.cc" litert_tensor_core LABELS internal)
litert_tensor_add_test(litert_tensor_internal_type_id_test "${LITERT_TENSOR_SOURCE_DIR}/internal/type_id_test.cc"
  litert_tensor_core LABELS internal)
litert_tensor_add_test(litert_tensor_internal_compile_test "${LITERT_TENSOR_SOURCE_DIR}/internal/compile_test.cc"
  litert_tensor_core LABELS internal)
litert_tensor_add_test(litert_tensor_internal_op_debugger_test
  "${LITERT_TENSOR_SOURCE_DIR}/internal/op_debugger_test.cc" litert_tensor_core LABELS internal debugger
  LIBRARIES absl::scoped_mock_log)
litert_tensor_add_test(litert_tensor_internal_op_debugger_enabled_test
  "${LITERT_TENSOR_SOURCE_DIR}/internal/op_debugger_test.cc" litert_tensor_core_debugger
  LABELS internal debugger debugger_enabled LIBRARIES absl::scoped_mock_log)
target_compile_definitions(litert_tensor_internal_op_debugger_test PRIVATE
                           LITERT_OP_DEBUGGER_ENABLED=1)
target_compile_definitions(litert_tensor_internal_op_debugger_enabled_test PRIVATE
                           LITERT_OP_DEBUGGER_ENABLED=1)

litert_tensor_add_test(litert_tensor_common_backend_utils_test
  "${LITERT_TENSOR_SOURCE_DIR}/backends/common_nnpack/utils_test.cc" litert_tensor_common_backend
  LABELS backends common_nnpack)
litert_tensor_add_test(litert_tensor_common_backend_conversion_test
  "${LITERT_TENSOR_SOURCE_DIR}/backends/common_nnpack/conversion_test.cc" litert_tensor_common_backend
  LABELS backends common_nnpack)
litert_tensor_add_test(litert_tensor_xnnpack_conversion_test
  "${LITERT_TENSOR_SOURCE_DIR}/backends/xnnpack/conversion_test.cc" litert_tensor_xnnpack_backend
  LABELS backends xnnpack)
litert_tensor_add_test(litert_tensor_xnnpack_arithmetic_test
  "${LITERT_TENSOR_SOURCE_DIR}/backends/xnnpack/arithmetic_test.cc" litert_tensor_xnnpack_runner
  LABELS backends xnnpack)
litert_tensor_add_test(litert_tensor_xnnpack_conversion_numerical_test
  "${LITERT_TENSOR_SOURCE_DIR}/backends/xnnpack/xnnpack_conversion_numerical_test.cc"
  litert_tensor_xnnpack_runner LABELS backends xnnpack numerical)

litert_tensor_add_test(litert_tensor_common_runner_test
  "${LITERT_TENSOR_SOURCE_DIR}/runners/common_nnpack/runner_test.cc" litert_tensor_common_runner
  LABELS runners common_nnpack)
litert_tensor_add_test(litert_tensor_xnnpack_runner_test
  "${LITERT_TENSOR_SOURCE_DIR}/runners/xnnpack/runner_test.cc" litert_tensor_xnnpack_runner
  LABELS runners xnnpack)
litert_tensor_add_test(litert_tensor_macros_test "${LITERT_TENSOR_SOURCE_DIR}/utils/macros_test.cc"
  litert_tensor_core LABELS utils)

option(LITERT_TENSOR_BUILD_GEMMA4 "Build Gemma4 model libraries and portable tests" ON)
option(LITERT_TENSOR_BUILD_NATIVE "Build the optimized matched-bundle Gemma4 runner" ON)
if(LITERT_TENSOR_BUILD_GEMMA4)
  include("${CMAKE_CURRENT_LIST_DIR}/examples.cmake")
endif()

get_property(manifest_entries GLOBAL PROPERTY LITERT_TENSOR_TEST_MANIFEST_ENTRIES)
list(JOIN manifest_entries ",\n" manifest_body)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tensor-tests.json"
  CONTENT "{\n  \"tests\": [\n${manifest_body}\n  ]\n}\n")
