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

# Samsung NPU Build Configuration
cmake_minimum_required(VERSION 3.20)

# This file is included from litert/vendors/CMakeLists.txt

message(STATUS "Configuring Samsung NPU support")

# Root-level libraries
# SOC Model library
add_library(samsung_soc_model STATIC
    soc_model.cc
)

target_include_directories(samsung_soc_model
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(samsung_soc_model
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
)

# AI LiteCore Manager library
add_library(samsung_ai_litecore_manager STATIC
    ai_litecore_manager.cc
)

target_include_directories(samsung_ai_litecore_manager
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${LITECORE_HEADERS_DIR}>:${LITECORE_HEADERS_DIR}>
)

target_link_libraries(samsung_ai_litecore_manager
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        litert_cc_internal
)

# Add subdirectories for modular build
add_subdirectory(compiler)
add_subdirectory(dispatch)
