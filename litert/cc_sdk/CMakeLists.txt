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

# LiteRT C++ API
cmake_minimum_required(VERSION 3.20)

project(litert_cc_api LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LITERT_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
set(LITERT_BUILD_COMMON_GENERATED_DIR "${LITERT_GENERATED_INCLUDE_DIR}/litert/build_common")
file(MAKE_DIRECTORY "${LITERT_BUILD_COMMON_GENERATED_DIR}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/litert/build_common/build_config.h.in
    ${LITERT_BUILD_COMMON_GENERATED_DIR}/build_config.h
)
include_directories(${LITERT_GENERATED_INCLUDE_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(absl REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(opencl_headers REQUIRED)

# C API library
add_library(litert_c_api SHARED IMPORTED)
set_target_properties(litert_c_api PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.so"
)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/litert_cc_sdk_hdrs.txt FILE_CONTENTS)
string(REGEX REPLACE "[ \t\r\n]+" ";" LITERT_CC_HEADERS "${FILE_CONTENTS}")
list(REMOVE_ITEM LITERT_CC_HEADERS "")

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/litert_cc_sdk_srcs.txt FILE_CONTENTS)
string(REGEX REPLACE "[ \t\r\n]+" ";" LITERT_CC_SRCS "${FILE_CONTENTS}")
list(REMOVE_ITEM LITERT_CC_SRCS "")

# Core C++ API library
add_library(litert_cc_api STATIC
    ${LITERT_CC_SRCS}
)

# Set PUBLIC_HEADER property for proper installation
set_target_properties(litert_cc_api PROPERTIES
    PUBLIC_HEADER "${LITERT_CC_HEADERS}"
)

target_include_directories(litert_cc_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${LITERT_GENERATED_INCLUDE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_cc_api
    PUBLIC
        absl::status
        absl::statusor
        absl::span
        absl::strings
        absl::log
        flatbuffers::flatbuffers
    PRIVATE
        litert_c_api
)

# run_model_simple target for testing.
add_executable(run_model_simple
    litert/tools/run_model_simple.cc
)

target_include_directories(run_model_simple
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(run_model_simple
    PRIVATE
        litert_cc_api
        absl::flags
        absl::flags_parse
        absl::random_random
        absl::log
        absl::log_internal_check_op
        absl::string_view
        absl::span
)
