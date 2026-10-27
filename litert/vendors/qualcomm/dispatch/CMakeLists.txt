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
    target_link_libraries(dispatch_api_qualcomm_so PRIVATE log)
endif()

set_target_properties(dispatch_api_qualcomm_so PROPERTIES
    OUTPUT_NAME "LiteRtDispatch_Qualcomm"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN"
)

if(WIN32)
    # The runtime discovers dispatch libraries by the "libLiteRtDispatch" name
    # prefix (litert/core/dynamic_loading.h, FindLiteRtDispatchSharedLibs), but
    # CMAKE_SHARED_LIBRARY_PREFIX is empty on Windows, so the default output is
    # LiteRtDispatch_Qualcomm.dll and is never found. Keep the "lib" prefix the
    # other platforms get for free.
    set_target_properties(dispatch_api_qualcomm_so PROPERTIES PREFIX "lib")

    # See litert/c/litert_common.h: without this the C API entry points this
    # library defines are declared __declspec(dllimport) and MSVC rejects the
    # definitions (C2491). The Bazel build passes the same define.
    target_compile_definitions(dispatch_api_qualcomm_so PRIVATE
        LITERT_COMPILE_LIBRARY)

    # Windows has no rpath; LiteRt.dll is found via the loader search path,
    # which includes the directory of the module being loaded.
    target_link_libraries(dispatch_api_qualcomm_so PRIVATE
        litert_runtime_c_api_shared_lib
        litert_compiler_plugin
        litert_cc_api
        litert_cc_options
        tensorflow-lite
        Threads::Threads
    )
elseif(UNIX AND NOT APPLE)
    # Set soname and rpath to search origin directory for libLiteRt.so
    target_link_options(dispatch_api_qualcomm_so PRIVATE
        "-Wl,-soname=libLiteRtDispatch_Qualcomm.so"
        "-Wl,-rpath,\$ORIGIN"
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

    target_link_options(dispatch_api_qualcomm_so PRIVATE
        "-Wl,--version-script=${PROJECT_SOURCE_DIR}/build_common/export_litert_only_linux.lds")
    set_target_properties(dispatch_api_qualcomm_so PROPERTIES
        LINK_DEPENDS "${PROJECT_SOURCE_DIR}/build_common/export_litert_only_linux.lds")
endif()
