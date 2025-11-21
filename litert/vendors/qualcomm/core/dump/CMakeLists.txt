# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# QNN Dump Utilities Library
cmake_minimum_required(VERSION 3.20)

# Create dump library with explicit source list
add_library(qnn_dump STATIC)

target_sources(qnn_dump
    PRIVATE
        dump_graph.cc
)

target_include_directories(qnn_dump
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(qnn_dump
    PUBLIC
        qnn_core
        absl::status
        absl::statusor
        nlohmann_json::nlohmann_json
)
