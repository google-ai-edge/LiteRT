# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm Compiler Plugin Shared Library
cmake_minimum_required(VERSION 3.20)

message(STATUS "Building Qualcomm compiler plugin for host")

# Dependencies (nlohmann_json, googletest, googlebenchmark) are fetched at parent level

# Create compiler plugin shared library
add_library(qnn_compiler_plugin SHARED
    qnn_compiler_plugin.cc
    qnn_compose_graph.cc
    graph_mapper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cc/namespace_heuristics.cc
)

target_include_directories(qnn_compiler_plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(qnn_compiler_plugin
    PUBLIC
        qnn_core
    PRIVATE
        qnn_builders
        qnn_manager
        qnn_transformation
        qnn_dump
        qnn_context_binary_info
        tensorflow-lite
        absl::log
        absl::check
)

set_target_properties(qnn_compiler_plugin PROPERTIES
    OUTPUT_NAME "LiteRtCompilerPlugin_Qualcomm"
)

if(UNIX AND NOT APPLE)
    target_link_options(qnn_compiler_plugin PRIVATE 
        "-Wl,-soname=libLiteRtCompilerPlugin_Qualcomm.so"
    )

    target_link_libraries(qnn_compiler_plugin PRIVATE 
        "-Wl,--start-group"
        litert_runtime_c_api_shared_lib 
        tensorflow-lite
        Threads::Threads
        litert_compiler_plugin
        litert_cc_api
        litert_cc_options
        "-Wl,--end-group"
    )
endif()
