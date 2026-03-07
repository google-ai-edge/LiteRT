# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm QNN Core Library
cmake_minimum_required(VERSION 3.20)

add_library(qnn_soc_table STATIC)

target_sources(qnn_soc_table PRIVATE soc_table.cc)

target_include_directories(qnn_soc_table
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
