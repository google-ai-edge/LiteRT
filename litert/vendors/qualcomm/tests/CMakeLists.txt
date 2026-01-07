# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm QNN Tests
cmake_minimum_required(VERSION 3.20)

# Common test library
add_library(litert_test_common STATIC)

target_sources(litert_test_common
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../test/common.cc
)

target_include_directories(litert_test_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../test
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${TENSORFLOW_SOURCE_DIR}
    ${TENSORFLOW_SOURCE_DIR}/third_party/xla/third_party/tsl
)

target_link_libraries(litert_test_common PUBLIC
    GTest::gmock
    tensorflow-lite
    absl::string_view
    flatbuffers::flatbuffers
)

# Compiler plugin test
add_executable(qnn_compiler_plugin_test)

target_sources(qnn_compiler_plugin_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../compiler/qnn_compiler_plugin_test.cc
)

target_link_libraries(qnn_compiler_plugin_test
    PRIVATE
        qnn_compiler_plugin
        litert_test_common
        litert_runtime_c_api_shared_lib 
        GTest::gtest_main
)

# QNN manager test
add_executable(qnn_manager_test)

target_sources(qnn_manager_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../qnn_manager_test.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/dump.cc
)

target_link_libraries(qnn_manager_test
    PRIVATE
        absl::log_internal_message
        qnn_compiler_plugin
        litert_test_common
        GTest::gtest_main
)

# Core common test
add_executable(common_test)

target_sources(common_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/common_test.cc
)
target_link_libraries(common_test
    PRIVATE
        qnn_core
        GTest::gtest_main
)

# Tensor pool test
add_executable(tensor_pool_test)

target_sources(tensor_pool_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/tensor_pool_test.cc
)

target_link_libraries(tensor_pool_test
    PRIVATE
        qnn_wrappers
        qnn_core
        GTest::gtest_main
)

# Op wrapper test
add_executable(op_wrapper_test)

target_sources(op_wrapper_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/wrappers/tests/op_wrapper_test.cc
)

target_link_libraries(op_wrapper_test
    PRIVATE
        qnn_wrappers
        qnn_core
        GTest::gtest_main
)

# Tensor wrapper test
add_executable(tensor_wrapper_test)

target_sources(tensor_wrapper_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/wrappers/tests/tensor_wrapper_test.cc
)

target_link_libraries(tensor_wrapper_test
    PRIVATE
        qnn_wrappers
        qnn_core
        GTest::gmock
        GTest::gtest_main
)

# Param wrapper test
add_executable(param_wrapper_test)

target_sources(param_wrapper_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/wrappers/tests/param_wrapper_test.cc
)

target_link_libraries(param_wrapper_test
    PRIVATE
        qnn_wrappers
        qnn_core
        GTest::gtest_main
)

# Quantize params wrapper test
add_executable(quantize_params_wrapper_test)

target_sources(quantize_params_wrapper_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/wrappers/tests/quantize_params_wrapper_test.cc
)

target_link_libraries(quantize_params_wrapper_test
    PRIVATE
        qnn_wrappers
        qnn_core
        GTest::gtest_main
)

# Utils test
add_executable(utils_test)

target_sources(utils_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/utils_test.cc
)

target_link_libraries(utils_test
    PRIVATE
        qnn_backends
        qnn_core
        GTest::gtest_main
)

# IR backend test
add_executable(ir_backend_test)

target_sources(ir_backend_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/backends/ir_backend_test.cc
)

target_link_libraries(ir_backend_test
    PRIVATE
        qnn_backends
        qnn_core
        GTest::gtest_main
)

# HTP backend test
add_executable(htp_backend_test)

target_sources(htp_backend_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/backends/htp_backend_test.cc
)

target_link_libraries(htp_backend_test
    PRIVATE
        qnn_backends
        qnn_core
        GTest::gtest_main
)

# DSP backend test
add_executable(dsp_backend_test)

target_sources(dsp_backend_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/backends/dsp_backend_test.cc
)

target_link_libraries(dsp_backend_test
    PRIVATE
        qnn_backends
        qnn_core
        GTest::gtest_main
)

# Backend utils test
add_executable(backend_utils_test)

target_sources(backend_utils_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/backends/backend_utils_test.cc
)

target_link_libraries(backend_utils_test
    PRIVATE
        qnn_backends
        qnn_core
        GTest::gtest_main
)
