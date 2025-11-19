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

# LiteRT Core Model
cmake_minimum_required(VERSION 3.20)

# Model library
add_library(litert_core_model STATIC
    flatbuffer_to_litert.cc
    graph_validation.cc
    litert_to_flatbuffer.cc
    model.cc
    model_buffer.cc
    model_file_test_util.cc
    model_load.cc
    model_serialize.cc
)

target_include_directories(litert_core_model
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_core_model
    PUBLIC
        litert_core
        absl::status
        absl::statusor
        absl::strings
        absl::span
        flatbuffers::flatbuffers
    PRIVATE
        litert_c_api
)

# We need the schema_generated.h from TFLite
target_include_directories(litert_core_model
    PRIVATE
        ${TFLITE_SOURCE_DIR}
)

set(_tflite_schema_dir "${TFLITE_SOURCE_DIR}/converter/schema")
set(_mutable_schema_out_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include/tflite/schema/mutable")
set(_mutable_schema_hdr "${_mutable_schema_out_dir}/schema_generated.h")

if(NOT EXISTS "${_tflite_schema_dir}/schema.fbs")
  message(FATAL_ERROR "TFLite converter schema.fbs not found at ${_tflite_schema_dir}")
endif()

set(FLATC_EXECUTABLE "")
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  if(TFLITE_HOST_TOOLS_DIR AND EXISTS "${TFLITE_HOST_TOOLS_DIR}/flatc")
    set(FLATC_EXECUTABLE "${TFLITE_HOST_TOOLS_DIR}/flatc")
  elseif(TFLITE_BUILD_DIR AND EXISTS "${TFLITE_BUILD_DIR}/host_flatc/_deps/flatbuffers-build/flatc")
    set(FLATC_EXECUTABLE "${TFLITE_BUILD_DIR}/host_flatc/_deps/flatbuffers-build/flatc")
  endif()
endif()
if(NOT FLATC_EXECUTABLE)
  find_program(FLATC_EXECUTABLE NAMES flatc)
endif()
if(NOT FLATC_EXECUTABLE)
  include(FetchContent)
  message(STATUS "flatc not found; fetching FlatBuffers to build flatc")
  FetchContent_Declare(flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v25.9.23)
  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(flatbuffers)
  if(NOT TARGET flatc)
    message(FATAL_ERROR "FlatBuffers 'flatc' target not available after fetch")
  endif()
  set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
endif()

add_custom_command(
  OUTPUT "${_mutable_schema_hdr}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_mutable_schema_out_dir}"
  COMMAND ${FLATC_EXECUTABLE} --cpp --gen-mutable --gen-object-api -I "${_tflite_schema_dir}" -o "${_mutable_schema_out_dir}" "${_tflite_schema_dir}/schema.fbs"
  DEPENDS "${_tflite_schema_dir}/schema.fbs"
  COMMENT "Generating TFLite mutable schema (schema_generated.h) with flatc"
  VERBATIM)

add_custom_target(tflite_mutable_schema_gen DEPENDS "${_mutable_schema_hdr}")
add_dependencies(litert_core_model tflite_mutable_schema_gen)

target_include_directories(litert_core_model
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
