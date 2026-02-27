# Copyright 2026 Google LLC.
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

# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.20)
project(tflite_minimal C CXX)
set(CMAKE_CXX_STANDARD 17)

set(LITERT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../litert" CACHE PATH "Path to LiteRT source directory")
set(LITERT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_build" CACHE PATH "Path to LiteRT build directory")
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite" CACHE PATH "Path to TFLite source directory")

add_subdirectory(
    "${LITERT_SOURCE_DIR}" "${LITERT_BUILD_DIR}" EXCLUDE_FROM_ALL
)

add_executable(run_model_simple
    run_model_simple.cc
)

target_include_directories(run_model_simple
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Common libraries for tools that use TFLite
set(LITERT_LIBS
    litert_c_api
    litert_cc_api
    litert_cc_internal
    litert_cc_options
    litert_compiler_plugin
    litert_core
    litert_core_model
    litert_runtime
)

target_link_libraries(run_model_simple
    "-Wl,--start-group"
    absl::die_if_null
    absl::flags
    absl::flags_parse
    absl::log
    absl::log_internal_check_op
    absl::log_internal_message
    absl::string_view
    absl::span
    absl::status
    absl::statusor
    ${LITERT_LIBS}
    "-Wl,--end-group"
 )

add_executable(tflite_minimal
    tflite_minimal.cc
)

target_link_libraries(tflite_minimal
    tensorflow-lite
)
