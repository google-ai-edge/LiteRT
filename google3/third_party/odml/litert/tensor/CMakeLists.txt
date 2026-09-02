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

cmake_minimum_required(VERSION 3.20)

add_library(litert_tensor_api_utils INTERFACE)

target_include_directories(litert_tensor_api_utils
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/core/model/generated/include>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_tensor_api_utils
    INTERFACE
        litert_cc_internal
        absl::log
        absl::status
        absl::statusor
)

add_library(litert_tensor_api_internal STATIC
    internal/graph.cc
    internal/graph_probe.cc
    internal/graph_traversal.cc
    internal/shape.cc
)

target_include_directories(litert_tensor_api_internal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/core/model/generated/include>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_tensor_api_internal
    PUBLIC
        litert_tensor_api_utils
        litert_cc_internal
        absl::flat_hash_map
        absl::flat_hash_set
        absl::log
        absl::status
        absl::statusor
        absl::strings
        absl::string_view
        absl::span
)

add_library(litert_tensor_api_core STATIC
    tensor.cc
    buffer.cc
)

target_include_directories(litert_tensor_api_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/core/model/generated/include>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_tensor_api_core
    PUBLIC
        litert_tensor_api_internal
        litert_tensor_api_utils
        absl::check
        absl::log
        absl::status
        absl::statusor
        absl::strings
        absl::string_view
        absl::span
)

add_library(litert_tensor_api_tflite_backend STATIC
    backends/tflite/arithmetic_tflite.cc
    backends/tflite/tflite_flatbuffer_conversion.cc
)

target_include_directories(litert_tensor_api_tflite_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/core/model/generated/include>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_tensor_api_tflite_backend
    PUBLIC
        litert_tensor_api_core
        litert_tensor_api_internal
        litert_tensor_api_utils
        tensorflow-lite
        flatbuffers::flatbuffers
        absl::btree
        absl::log
        absl::status
        absl::statusor
        absl::strings
        absl::string_view
        absl::span
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/segmentation/CMakeLists.txt")
    add_subdirectory(examples/segmentation)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/gemma3/CMakeLists.txt")
    add_subdirectory(examples/gemma3)
endif()