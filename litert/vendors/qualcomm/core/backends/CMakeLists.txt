# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# QNN Backends Library
cmake_minimum_required(VERSION 3.20)

# Create backends library with explicit source list
add_library(qnn_backends STATIC)

target_sources(qnn_backends
    PRIVATE
        dsp_backend.cc
        htp_backend.cc
        ir_backend.cc
        qnn_backend.cc
)

target_include_directories(qnn_backends
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_backends
    PRIVATE
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
        absl::span
        absl::string_view
)
