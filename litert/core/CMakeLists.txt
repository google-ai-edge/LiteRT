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

# LiteRT Core
cmake_minimum_required(VERSION 3.20)

# Core library
add_library(litert_core STATIC
    build_stamp.cc
    buffer_error_reporter.cc
    error_reporter.cc
    dispatch_op_schema.cc
    dynamic_loading.cc
    dynamic_loading_windows.cc
    environment.cc
    environment_options.cc
    filesystem.cc
    litert_toml_parser.cc
    util/flatbuffer_tools.cc
    util/tensor_type_util.cc
)

target_include_directories(litert_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_core
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        absl::flat_hash_set
        flatbuffers::flatbuffers
        litert_logging
    PRIVATE
        Threads::Threads
)

# Platform-specific source exclusions
if(NOT WIN32)
    # Exclude Windows-specific source
    set_source_files_properties(dynamic_loading_windows.cc PROPERTIES HEADER_FILE_ONLY TRUE)
endif()

# Add model subdirectory
add_subdirectory(model)
add_subdirectory(cache)
