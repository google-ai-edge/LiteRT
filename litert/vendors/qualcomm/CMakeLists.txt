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

# Qualcomm QNN Build Configuration
cmake_minimum_required(VERSION 3.20)

# This file is included from litert/vendors/CMakeLists.txt
# Variables available: QAIRT_HEADERS_DIR, TFLITE_BUILD_DIR, etc.

message(STATUS "Configuring Qualcomm QNN support")

# Fetch dependencies needed by core libraries (dump needs nlohmann_json)
include(FetchContent)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.12.x
    )
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG 0d98dba29d66e93259db7daa53a9327df767a415 # v1.6.1
)

FetchContent_MakeAvailable(nlohmann_json)
FetchContent_MakeAvailable(googletest)

# Disable -Werror for googlebenchmark to avoid -Wunused-but-set-variable errors
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Root-level libraries (used by core and other components)
# QNN Manager library
add_library(qnn_manager STATIC
    qnn_manager.cc
)

target_include_directories(qnn_manager
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_manager
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
)

# Context Binary Info library
add_library(qnn_context_binary_info STATIC
    context_binary_info.cc
)

target_include_directories(qnn_context_binary_info
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_context_binary_info
    PUBLIC
        absl::status
        absl::statusor
)

# Add subdirectories for modular build
add_subdirectory(core)
add_subdirectory(dispatch)
add_subdirectory(compiler)

# Optionally add tests
if(LITERT_BUILD_TESTS)
    add_subdirectory(tests)
endif()
