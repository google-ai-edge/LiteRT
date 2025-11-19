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

# Google Tensor flags
add_library(litert_tool_flags_google_tensor STATIC
    google_tensor_flags.cc
)

target_include_directories(litert_tool_flags_google_tensor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tool_flags_google_tensor
    PUBLIC
        absl::flags
)

# MediaTek flags
add_library(litert_tool_flags_mediatek STATIC
    mediatek_flags.cc
)

target_include_directories(litert_tool_flags_mediatek
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tool_flags_mediatek
    PUBLIC
        absl::flags
)

# Qualcomm flags
add_library(litert_tool_flags_qualcomm STATIC
    qualcomm_flags.cc
)

target_include_directories(litert_tool_flags_qualcomm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tool_flags_qualcomm
    PUBLIC
        absl::flags
)

# Intel OpenVINO flags
add_library(litert_tool_flags_intel_openvino STATIC
    intel_openvino_flags.cc
)

target_include_directories(litert_tool_flags_intel_openvino
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tool_flags_intel_openvino
    PUBLIC
        absl::flags
)
