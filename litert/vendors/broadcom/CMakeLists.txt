# Copyright (c) 2026 Broadcom. All Rights Reserved.
# The term "Broadcom" refers to Broadcom Inc. and/or its subsidiaries.
# This software is owned by Broadcom and licensed under Apache-2.0.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

message(STATUS "Configuring Broadcom BNE support")

set(_bstorm_so "${BSTORM_PREBUILT_DIR}/libbstorm_core.so")
if(NOT EXISTS "${_bstorm_so}")
  message(FATAL_ERROR
    "libbstorm_core.so not found at ${_bstorm_so}.\n"
    "  Supply via -DBSTORM_PREBUILT_DIR=<dir> or -DBSTORM_PREBUILT_URL=<url>")
endif()

if(BSTORM_HEADERS_DIR AND EXISTS "${BSTORM_HEADERS_DIR}")
  set(_bstorm_inc "${BSTORM_HEADERS_DIR}")
else()
  set(_bstorm_inc "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if(NOT EXISTS "${_bstorm_inc}/bstorm_init.h")
    message(FATAL_ERROR
      "BSTORM headers not found.\n"
      "  Supply via -DBSTORM_HEADERS_DIR=<dir> or -DBSTORM_HEADERS_URL=<url>")
  endif()
endif()

add_library(bstorm_core_shared SHARED IMPORTED GLOBAL)
set_target_properties(bstorm_core_shared PROPERTIES
  IMPORTED_LOCATION "${_bstorm_so}")
target_include_directories(bstorm_core_shared INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${_bstorm_inc}
  ${_bstorm_inc}/core
  ${_bstorm_inc}/src
)

add_subdirectory(common)
add_subdirectory(compiler)
add_subdirectory(dispatch)
add_subdirectory(example)

if(UNIX AND NOT APPLE)
  target_link_options(bstorm_dispatch_plugin PRIVATE
    "-Wl,-soname=libLiteRtDispatch_bstorm.so"
    "-Wl,-rpath,$ORIGIN"
  )
endif()

target_link_libraries(litert_cc_api PUBLIC flatbuffers absl::log_internal_check_op)
