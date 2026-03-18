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

# LiteRT C API Options
cmake_minimum_required(VERSION 3.20)

# Compatibility target: source-build CMake now compiles all public c/options
# sources directly into litert_c_api so consumers only need the main C API
# library.
add_library(litert_c_options INTERFACE)

target_include_directories(litert_c_options
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(litert_c_options
    INTERFACE
        litert_c_api
        absl::status
        absl::statusor
)
