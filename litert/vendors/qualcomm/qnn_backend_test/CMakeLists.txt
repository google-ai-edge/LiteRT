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
        qnn_backends
    PRIVATE
        GTest::gtest
)

# Register QnnModelTest parameters once per test executable. An object library
# lets the CMake build compile the registration source once, then reuse it in
# both the aggregate builder test and qnn_model_test binaries.
add_library(qnn_model_test_instantiation OBJECT)

target_sources(qnn_model_test_instantiation
    PRIVATE
        qnn_model_test_instantiation.cc
)

target_link_libraries(qnn_model_test_instantiation
    PRIVATE
        qnn_test_utils
        GTest::gtest
)
