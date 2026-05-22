# Copyright (C) 2026 Samsung Electronics Co. LTD.
# SPDX-License-Identifier: Apache-2.0
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
# Samsung Dispatch API Shared Library
cmake_minimum_required(VERSION 3.20)

set(FLATC_EXECUTABLE "")

if(TFLITE_HOST_TOOLS_DIR AND EXISTS "${TFLITE_HOST_TOOLS_DIR}/flatc")
  set(FLATC_EXECUTABLE "${TFLITE_HOST_TOOLS_DIR}/flatc")
elseif(TFLITE_BUILD_DIR AND EXISTS "${TFLITE_BUILD_DIR}/host_flatc/_deps/flatbuffers-build/flatc")
  set(FLATC_EXECUTABLE "${TFLITE_BUILD_DIR}/host_flatc/_deps/flatbuffers-build/flatc")
elseif(TFLITE_BUILD_DIR AND EXISTS "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/flatc")
  set(FLATC_EXECUTABLE "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/flatc")
endif()

if(NOT FLATC_EXECUTABLE)
  find_program(FLATC_EXECUTABLE NAMES flatc)
endif()

if(NOT FLATC_EXECUTABLE)
  message(STATUS "flatc not found in PATH; fetching FlatBuffers to build flatc")
  include(FetchContent)
  FetchContent_Declare(flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v25.9.23
  )
  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(flatbuffers)
  if(NOT TARGET flatc)
    message(FATAL_ERROR "FlatBuffers 'flatc' target not available after fetch")
  endif()
  set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
endif()

# Generate FlatBuffer C++ header from schema
set(_samsung_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include/litert/vendors/samsung/schema")
file(MAKE_DIRECTORY "${_samsung_gen_dir}")

set(_samsung_schema "${CMAKE_CURRENT_SOURCE_DIR}/litert_samsung_header.fbs")
set(_samsung_gen_hdr "${_samsung_gen_dir}/litert_samsung_header_generated.h")

add_custom_command(
  OUTPUT "${_samsung_gen_hdr}"
  COMMAND ${FLATC_EXECUTABLE} --cpp --gen-mutable --gen-object-api -o "${_samsung_gen_dir}" "${_samsung_schema}"
  DEPENDS "${_samsung_schema}"
  COMMENT "Generating Samsung litert_samsung_header_generated.h with flatc"
  VERBATIM
)

add_custom_target(samsung_schema_gen DEPENDS "${_samsung_gen_hdr}")

# Samsung byte code library
add_library(samsung_byte_code STATIC
    samsung_byte_code.cc
)

add_dependencies(samsung_byte_code samsung_schema_gen)

target_include_directories(samsung_byte_code
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>
)

target_link_libraries(samsung_byte_code
    PUBLIC
        flatbuffers::flatbuffers
        absl::span
        absl::strings
    PRIVATE
        litert_cc_internal
)

# Byte code print tool
add_executable(byte_code_print byte_code_print.cc)

add_dependencies(byte_code_print samsung_schema_gen)

target_include_directories(byte_code_print
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>
)

target_link_libraries(byte_code_print
    PRIVATE
        samsung_byte_code
        absl::flags
        absl::flags_parse
)
