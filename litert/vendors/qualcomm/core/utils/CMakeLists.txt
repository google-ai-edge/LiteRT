# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm QNN Core Library
cmake_minimum_required(VERSION 3.20)

# log
add_library(qnn_log STATIC)

# Platform-specific logging
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_sources(qnn_log PRIVATE log_android.cc)
else()
    target_sources(qnn_log PRIVATE log_default.cc)
endif()

target_include_directories(qnn_log
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_log 
    PRIVATE
        qnn_common
) 

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(qnn_log PUBLIC android log)
endif()

# miscs
add_library(qnn_miscs STATIC)

target_sources(qnn_miscs PRIVATE miscs.cc)

target_include_directories(qnn_miscs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
)

target_link_libraries(qnn_miscs
    PUBLIC
        qnn_log
        qnn_soc_table
        absl::span
)
