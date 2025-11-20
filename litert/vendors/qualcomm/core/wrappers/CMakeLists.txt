# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# QNN Wrappers Library
cmake_minimum_required(VERSION 3.20)

# Create wrappers library with explicit source list
add_library(qnn_wrappers STATIC)

target_sources(qnn_wrappers
    PRIVATE
        op_wrapper.cc
        param_wrapper.cc
        quantize_params_wrapper.cc
        tensor_wrapper.cc
)

target_include_directories(qnn_wrappers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_wrappers
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
        absl::span
        absl::string_view
)
