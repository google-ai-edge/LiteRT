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
)

target_link_libraries(samsung_ai_litecore_manager
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        litert_cc_internal
)

# Compiler plugin
add_library(samsung_compiler_plugin STATIC
    compiler/compiler_plugin.cc
    compiler/compile_model.cc
    compiler/create_model.cc
    compiler/builders/batch_matmul_op_builder.cc
    compiler/builders/cast_op_builder.cc
    compiler/builders/concat_op_builder.cc
    compiler/builders/conv2d_op_builder.cc
    compiler/builders/cumsum_op_builder.cc
    compiler/builders/elementwise_op_builder.cc
    compiler/builders/fully_connected_op_builder.cc
    compiler/builders/gather_op_builder.cc
    compiler/builders/gathernd_op_builder.cc
    compiler/builders/gelu_op_builder.cc
    compiler/builders/hardswish_op_builder.cc
    compiler/builders/leakyrelu_op_builder.cc
    compiler/builders/logistic_op_builder.cc
    compiler/builders/pad_op_builder.cc
    compiler/builders/pool2d_op_builder.cc
    compiler/builders/reduce_op_builder.cc
    compiler/builders/relu_op_builder.cc
    compiler/builders/reshape_op_builder.cc
    compiler/builders/resizebilinear_op_builder.cc
    compiler/builders/select_op_builder.cc
    compiler/builders/selectv2_op_builder.cc
    compiler/builders/slice_op_builder.cc
    compiler/builders/softmax_op_builder.cc
    compiler/builders/spacetodepth_op_builder.cc
    compiler/builders/split_op_builder.cc
    compiler/builders/strided_slice_op_builder.cc
    compiler/builders/tanh_op_builder.cc
    compiler/builders/transpose_conv_op_builder.cc
    compiler/builders/transpose_op_builder.cc
    compiler/builders/utils.cc
)

target_include_directories(samsung_compiler_plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compiler>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compiler/builders>
)

target_link_libraries(samsung_compiler_plugin
    PUBLIC
        litert_compiler_plugin
        litert_core
        litert_core_model
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
        samsung_soc_model
)

# Dispatch library
add_library(samsung_dispatch STATIC
    dispatch/dispatch_api.cc
    dispatch/enn_manager.cc
    dispatch/litert_dispatch_device_context.cc
    dispatch/litert_dispatch_invocation_context.cc
)

target_include_directories(samsung_dispatch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dispatch>
)

target_link_libraries(samsung_dispatch
    PUBLIC
        litert_runtime_c_api_static
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
        samsung_soc_model
        samsung_ai_litecore_manager
)
