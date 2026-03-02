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

# LiteRT C++ Internal Implementation
cmake_minimum_required(VERSION 3.20)

# Internal implementation library - header-only since no .cc files exist in this directory
add_library(litert_cc_internal INTERFACE)

target_include_directories(litert_cc_internal
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_cc_internal
    INTERFACE
        litert_c_api
        litert_cc_api
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        absl::synchronization
        flatbuffers::flatbuffers
        Threads::Threads
)

# Platform-specific configurations
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(litert_cc_internal INTERFACE ${LOG_LIB})
endif()
