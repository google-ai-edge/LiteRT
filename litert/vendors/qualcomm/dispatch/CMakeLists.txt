# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualcomm Dispatch API Shared Library
cmake_minimum_required(VERSION 3.20)

add_library(dispatch_api_qualcomm_so SHARED)

target_sources(dispatch_api_qualcomm_so
    PRIVATE
        dispatch_api.cc
        litert_dispatch_device_context.cc
        litert_dispatch_invocation_context.cc
)

target_include_directories(dispatch_api_qualcomm_so
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(dispatch_api_qualcomm_so
    PUBLIC
        qnn_core
    PRIVATE
        qnn_manager
        qnn_context_binary_info
        litert_runtime_c_api_static
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(dispatch_api_qualcomm_so PRIVATE android log)
endif()

set_target_properties(dispatch_api_qualcomm_so PROPERTIES
    OUTPUT_NAME "LiteRtDispatch_Qualcomm"
)

if(UNIX AND NOT APPLE)
    # Set soname
    target_link_options(dispatch_api_qualcomm_so PRIVATE 
        "-Wl,-soname=libLiteRtDispatch_Qualcomm.so"
    )

    target_link_libraries(dispatch_api_qualcomm_so PRIVATE 
        "-Wl,--start-group"
        litert_runtime_c_api_shared_lib 
        litert_compiler_plugin
        litert_cc_api
        litert_cc_options
        tensorflow-lite
        Threads::Threads
        "-Wl,--end-group"
    )
endif()
