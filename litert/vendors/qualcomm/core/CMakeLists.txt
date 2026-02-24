# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm QNN Core Library
cmake_minimum_required(VERSION 3.20)

# Common QNN library
add_library(qnn_common STATIC
    common.cc
)

target_include_directories(qnn_common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_common
    PUBLIC
        absl::string_view
        absl::strings
        absl::str_format
)

# tensor pool
add_library(qnn_tensor_pool STATIC)

target_sources(qnn_tensor_pool PRIVATE tensor_pool.cc)

target_link_libraries(qnn_tensor_pool
    PRIVATE
        qnn_log
        qnn_miscs
        qnn_wrappers
)

# Add subdirectories for backends, wrappers, builders, transformation, dump (used by compiler)
add_subdirectory(backends)
add_subdirectory(wrappers)
add_subdirectory(builders)
add_subdirectory(transformation)
add_subdirectory(dump)
add_subdirectory(utils)
add_subdirectory(schema)
