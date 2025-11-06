# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm QNN Core Library
cmake_minimum_required(VERSION 3.20)

# Core QNN library - shared between dispatch and compiler
add_library(qnn_core STATIC
    # Core
    common.cc
    tensor_pool.cc
    
    # Utils
    utils/miscs.cc
    
    # Schema
    schema/soc_table.cc
)

# Platform-specific logging
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_sources(qnn_core PRIVATE utils/log_android.cc)
else()
    target_sources(qnn_core PRIVATE utils/log_default.cc)
endif()

target_include_directories(qnn_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_core
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
        absl::span
        absl::string_view
        qnn_backends
        qnn_wrappers
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(qnn_core PUBLIC android log)
endif()

# Add subdirectories for backends, wrappers, builders, transformation, dump (used by compiler)
add_subdirectory(backends)
add_subdirectory(wrappers)
add_subdirectory(builders)
add_subdirectory(transformation)
add_subdirectory(dump)
