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

cmake_minimum_required(VERSION 3.20)

set(LITERT_CORE_CACHE_SOURCES
    compilation_cache.cc
    compilation_cache.h
    hash_util.h
)

add_library(litert_core_cache STATIC ${LITERT_CORE_CACHE_SOURCES})

target_include_directories(litert_core_cache
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(litert_core_cache
    PUBLIC
        litert_compiler_plugin
        litert_core
        litert_core_model
        litert_c_api
        litert_c_options
        litert_cc_api
        absl::strings
)
