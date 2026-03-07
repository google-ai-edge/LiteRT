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

# LiteRT Compiler
cmake_minimum_required(VERSION 3.20)

# Compiler plugin library
add_library(litert_compiler_plugin STATIC
    plugin/algo.cc
    plugin/compiler_plugin.cc
    plugin/litert_compiler_options.cc
)

target_include_directories(litert_compiler_plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_compiler_plugin
    PUBLIC
        litert_core
        litert_core_model
        litert_c_api
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        absl::flat_hash_set
        flatbuffers::flatbuffers
)
