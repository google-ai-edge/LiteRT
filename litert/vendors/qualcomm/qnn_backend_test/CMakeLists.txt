# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# QNN backend test utilities
cmake_minimum_required(VERSION 3.20)

add_library(qnn_test_utils STATIC)

target_sources(qnn_test_utils
    PRIVATE
        test_utils.cc
)

target_include_directories(qnn_test_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(qnn_test_utils
    PUBLIC
        qnn_common
        qnn_tensor_pool
        qnn_miscs
        qnn_model
        qnn_manager
    PRIVATE
        GTest::gtest
)
