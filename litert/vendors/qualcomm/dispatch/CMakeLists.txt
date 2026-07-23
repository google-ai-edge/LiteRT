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
    PRIVATE
        qnn_common
        qnn_wrappers
        qnn_miscs
        qnn_manager
        qnn_backends
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

if(LITERT_MAJOR_VERSION)
    set_target_properties(dispatch_api_qualcomm_so PROPERTIES
        VERSION   "${LITERT_MAJOR_VERSION}.${LITERT_MINOR_VERSION}.${LITERT_PATCH_VERSION}"
        SOVERSION "${LITERT_MAJOR_VERSION}"
    )
endif()

install(TARGETS dispatch_api_qualcomm_so
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT   Runtime
    NAMELINK_COMPONENT Development
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/litert_qcom_dispatch.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/litert_qcom_dispatch.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/litert_qcom_dispatch.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT   Development
)
