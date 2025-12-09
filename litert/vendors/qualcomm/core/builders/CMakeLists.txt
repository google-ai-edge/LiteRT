# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# QNN Op Builders Library
cmake_minimum_required(VERSION 3.20)

# Create builders library with explicit source list
# Using explicit list instead of file(GLOB) for better CMake practices:
# - CMake auto-reconfigures when this file changes
# - Explicit is better than implicit
# - Prevents accidental inclusion of temporary/test files
# - Better for code review and CI/CD
add_library(qnn_builders STATIC)

target_sources(qnn_builders
    PRIVATE
        arg_min_max_op_builder.cc
        broadcast_to_op_builder.cc
        cast_op_builder.cc
        concatenation_op_builder.cc
        conv2d_op_builder.cc
        conv3d_op_builder.cc
        cumsum_op_builder.cc
        depthwise_conv2d_op_builder.cc
        dynamic_update_slice_op_builder.cc
        elementwise_op_builder.cc
        embedding_lookup_op_builder.cc
        fully_connected_op_builder.cc
        fully_connected_op_builder_htp.cc
        gather_op_builder.cc
        gathernd_op_builder.cc
        gelu_op_builder.cc
        l2_norm_op_builder.cc
        leaky_relu_op_builder.cc
        logistic_op_builder.cc
        matmul_op_builder.cc
        op_builder.cc
        pack_op_builder.cc
        pad_op_builder.cc
        pool2d_op_builder.cc
        prelu_op_builder.cc
        quantize_op_builder.cc
        reduce_op_builder.cc
        relu_0to1_op_builder.cc
        relu_n1to1_op_builder.cc
        relu_op_builder.cc
        relu6_op_builder.cc
        reshape_op_builder.cc
        resize_op_builder.cc
        reverse_op_builder.cc
        rms_norm_op_builder.cc
        select_op_builder.cc
        slice_op_builder.cc
        softmax_op_builder.cc
        spatial_transform_op_builder.cc
        split_op_builder.cc
        strided_slice_op_builder.cc
        tanh_op_builder.cc
        tile_op_builder.cc
        transpose_conv_op_builder.cc
        transpose_op_builder.cc
        unpack_op_builder.cc
        log_softmax_op_builder.cc
        scatter_nd_op_builder.cc
        topk_op_builder.cc
        group_norm_op_builder.cc
)

target_include_directories(qnn_builders
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(qnn_builders
    PUBLIC
        qnn_core
)
