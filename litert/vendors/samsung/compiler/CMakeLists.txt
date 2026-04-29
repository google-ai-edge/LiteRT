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

# Samsung Compiler Plugin Shared Library
cmake_minimum_required(VERSION 3.20)

message(STATUS "Building Samsung compiler plugin for host")

# Create compiler plugin shared library
add_library(samsung_compiler_plugin SHARED
    compiler_plugin.cc
    compile_model.cc
    create_model.cc
    builders/argmax_op_builder.cc
    builders/batch_matmul_op_builder.cc
    builders/broadcastto_op_builder.cc
    builders/cast_op_builder.cc
    builders/concat_op_builder.cc
    builders/conv2d_op_builder.cc
    builders/conv3d_op_builder.cc
    builders/cumsum_op_builder.cc
    builders/dequantize_op_builder.cc
    builders/elementwise_op_builder.cc
    builders/fully_connected_op_builder.cc
    builders/gather_op_builder.cc
    builders/gathernd_op_builder.cc
    builders/gelu_op_builder.cc
    builders/hardswish_op_builder.cc
    builders/l2normalization_op_builder.cc
    builders/leakyrelu_op_builder.cc
    builders/logistic_op_builder.cc
    builders/pad_op_builder.cc
    builders/pool2d_op_builder.cc
    builders/quantize_op_builder.cc
    builders/reduce_op_builder.cc
    builders/relu_op_builder.cc
    builders/relu1_op_builder.cc
    builders/relu6_op_builder.cc
    builders/relun1to1_op_builder.cc
    builders/reshape_op_builder.cc
    builders/resizebilinear_op_builder.cc
    builders/resizenearestneighbor_op_builder.cc
    builders/reversev2_op_builder.cc
    builders/rms_norm_op_builder.cc
    builders/select_op_builder.cc
    builders/selectv2_op_builder.cc
    builders/slice_op_builder.cc
    builders/softmax_op_builder.cc
    builders/spacetodepth_op_builder.cc
    builders/split_op_builder.cc
    builders/strided_slice_op_builder.cc
    builders/tanh_op_builder.cc
    builders/transpose_conv_op_builder.cc
    builders/transpose_op_builder.cc
    builders/utils.cc
)

target_include_directories(samsung_compiler_plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    PRIVATE
        ${TFLITE_BUILD_DIR}/flatbuffers/include
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/builders>
        $<$<BOOL:${LITECORE_HEADERS_DIR}>:${LITECORE_HEADERS_DIR}>
)

target_link_libraries(samsung_compiler_plugin
    PRIVATE
        samsung_soc_model
        samsung_ai_litecore_manager
        tensorflow-lite
        absl::log
        absl::check
)

set_target_properties(samsung_compiler_plugin PROPERTIES
    OUTPUT_NAME "LiteRtCompilerPlugin_Samsung"
)

if(APPLE)
    target_link_libraries(samsung_compiler_plugin PRIVATE
        litert_runtime_c_api_shared_lib
        tensorflow-lite
        Threads::Threads
        litert_compiler_plugin
        litert_cc_api
        litert_cc_options
    )
elseif(UNIX)
    target_link_options(samsung_compiler_plugin PRIVATE
        "-Wl,-soname=libLiteRtCompilerPlugin_Samsung.so"
    )

    target_link_libraries(samsung_compiler_plugin PRIVATE
        "-Wl,--start-group"
        litert_runtime_c_api_shared_lib
        tensorflow-lite
        Threads::Threads
        litert_compiler_plugin
        litert_cc_api
        litert_cc_options
        "-Wl,--end-group"
    )
endif()
