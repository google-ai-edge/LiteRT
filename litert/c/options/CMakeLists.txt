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

# Options library
add_library(litert_c_options STATIC
    litert_darwinn_runtime_options.cc
    litert_cpu_options.cc
    litert_google_tensor_options.cc
    litert_gpu_options.cc
    litert_intel_openvino_options.cc
    litert_mediatek_options.cc
    litert_qualcomm_options.cc
    litert_runtime_options.cc
    litert_compiler_options.cc
)

target_include_directories(litert_c_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_c_options
    PUBLIC
        absl::status
        absl::statusor
    PRIVATE
        litert_c_api
)