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
    model_graph.cc
    model_load.cc
    # TODO (b/435736757): model_serialize.cc requires mutable schema which is
    # not generated in CMake build yet
    # model_serialize.cc
    model_serialize_stub.cc  # Temporary stub implementation
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
        litert_c_api
        absl::status
        absl::statusor
        absl::strings
        absl::span
        flatbuffers::flatbuffers
)

# We need the schema_generated.h from TFLite
target_include_directories(litert_core_model
    PRIVATE
        ${TFLITE_SOURCE_DIR}
)
