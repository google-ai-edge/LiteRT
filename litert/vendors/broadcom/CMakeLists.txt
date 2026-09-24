# Copyright (c) 2026 Broadcom. All Rights Reserved.
# The term "Broadcom" refers to Broadcom Inc. and/or its subsidiaries.
# This software is owned by Broadcom and licensed under Apache-2.0.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

message(STATUS "Configuring Broadcom BNE support")

set(LITERT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LITERT_REPO_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

if(NOT BSTORM_CORE_SOURCE_DIR)
  set(BSTORM_CORE_SOURCE_DIR "${LITERT_REPO_ROOT_DIR}/../bstorm/core")
endif()

if(NOT EXISTS "${BSTORM_CORE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "BSTORM core source not found at ${BSTORM_CORE_SOURCE_DIR}.\n"
    "  Supply via -DBSTORM_CORE_SOURCE_DIR=<dir>")
endif()

if(NOT DEFINED BSTORM_CHIP OR NOT DEFINED BSTORM_CHIP_REV OR NOT DEFINED BSTORM_BSTM_REV)
  message(FATAL_ERROR
    "BSTORM_CHIP, BSTORM_CHIP_REV, and BSTORM_BSTM_REV must all be set, e.g.\n"
    "  -DBSTORM_CHIP=74110 -DBSTORM_CHIP_REV=b0 -DBSTORM_BSTM_REV=2003")
endif()

add_subdirectory("${BSTORM_CORE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/core_build")

add_library(bstorm_core_shared SHARED "${LITERT_SOURCE_DIR}/c/empty.cc")
set_target_properties(bstorm_core_shared PROPERTIES OUTPUT_NAME bstorm_core)
target_include_directories(bstorm_core_shared PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LITERT_REPO_ROOT_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${BSTORM_CORE_SOURCE_DIR}/include
  ${BSTORM_CORE_SOURCE_DIR}/src/core
  ${BSTORM_CORE_SOURCE_DIR}/src
)
target_link_libraries(bstorm_core_shared PUBLIC bstorm_core_unified)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_subdirectory("${LITERT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/litert_build" EXCLUDE_FROM_ALL)
endif()

add_compile_options(
  -Wno-deprecated-declarations
  -Wno-nullability-completeness
  -Wno-return-type
  -fpermissive
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
