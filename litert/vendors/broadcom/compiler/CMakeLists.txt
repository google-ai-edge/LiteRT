# Copyright (c) 2026 Broadcom. All Rights Reserved.
# The term "Broadcom" refers to Broadcom Inc. and/or its subsidiaries.
# This software is owned by Broadcom and licensed under Apache-2.0.
#
# SPDX-License-Identifier: Apache-2.0

# Broadcom BSTORM Compiler Plugin Shared Library
cmake_minimum_required(VERSION 3.20)

add_library(bstorm_compiler_plugin SHARED
  bstorm_compiler_graph.cc
  bstorm_compiler_operation.cc
  bstorm_compiler_plugin.cc
)

set_target_properties(bstorm_compiler_plugin PROPERTIES
  OUTPUT_NAME LiteRtCompilerPlugin_bstorm
)

target_link_libraries(bstorm_compiler_plugin PRIVATE
  bstorm_core_shared
  bstorm_util_litert
)

if(UNIX AND NOT APPLE)
  target_link_options(bstorm_compiler_plugin PRIVATE
    "-Wl,-soname=libLiteRtCompilerPlugin_bstorm.so"
    "-Wl,-rpath,$ORIGIN"
  )
  target_link_libraries(bstorm_compiler_plugin PRIVATE
    "-Wl,--start-group"
    litert_cc_api
    litert_runtime_c_api_shared_lib
    "-Wl,--end-group"
    dl
  )
endif()
