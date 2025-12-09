# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# QNN Graph Transformation Library
cmake_minimum_required(VERSION 3.20)

add_library(qnn_transformation STATIC)

target_sources(qnn_transformation
    PRIVATE
        graph_to_graph.cc
        matmul_convert.cc
        embedding_gemma.cc
        mask.cc
        mha_to_sha.cc
)

target_include_directories(qnn_transformation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(qnn_transformation
    PUBLIC
        qnn_core
)
