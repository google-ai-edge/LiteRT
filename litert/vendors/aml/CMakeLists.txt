# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

message(STATUS "Configuring AML LiteRT vendor support")

set(AML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(AML_CORE_DIR "${AML_ROOT}/core")
set(AML_DISPATCH_DIR "${AML_ROOT}/dispatch")
set(AML_COMPILER_DIR "${AML_ROOT}/compiler")
get_filename_component(AML_REPO_ROOT "${AML_ROOT}/../../.." ABSOLUTE)

# Prebuilt libaml_compiler_core.so + nnsdk headers live in the external
# aml_npu repo (not inside this LiteRT tree).
# Default: <01_LiteRT>/aml_npu, sibling of this LiteRT checkout.
if(NOT AML_NPU_ROOT)
  if(DEFINED ENV{AML_NPU_ROOT} AND NOT "$ENV{AML_NPU_ROOT}" STREQUAL "")
    set(AML_NPU_ROOT "$ENV{AML_NPU_ROOT}")
  else()
    get_filename_component(AML_NPU_ROOT "${AML_REPO_ROOT}/../aml_npu" ABSOLUTE)
  endif()
endif()
set(AML_NPU_ROOT "${AML_NPU_ROOT}" CACHE PATH
  "External aml_npu repository (compiler_core + runtime_lib)" FORCE)
if(NOT EXISTS "${AML_NPU_ROOT}/compiler_core/cmake/AmlCompilerCorePrebuilt.cmake")
  message(FATAL_ERROR
    "External aml_npu not found at: ${AML_NPU_ROOT}\n"
    "  Pass -DAML_NPU_ROOT=/path/to/aml_npu")
endif()
include("${AML_NPU_ROOT}/compiler_core/cmake/AmlCompilerCorePrebuilt.cmake")
include("${AML_NPU_ROOT}/runtime_lib/cmake/AmlNpuRuntime.cmake")

set(AML_LINUX_LIB_DIR "${AML_NPU_RUNTIME_LIB_DIR}" CACHE PATH
  "Directory with Linux ARM prebuilt libs (libcompiler_adla.so, libnnsdk.so)")

# TFLite FlatBuffers v25 must precede vendor FlatBuffers if any leftover path leaks.
set(AML_COMMON_INCLUDES
  ${TFLITE_BUILD_DIR}/flatbuffers/include
  ${AML_COMPILER_CORE_INCLUDE_DIR}
  ${AML_CORE_DIR}
  ${AML_NNSDK_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/..
  ${TENSORFLOW_SOURCE_DIR}
  ${TFLITE_SOURCE_DIR}
)

# Packager format shared by compiler plugin + dispatch (no ADLA).
# Lives in vendors/aml/core (same as Bazel //litert/vendors/aml/core:aml_dispatch_info).
add_library(aml_dispatch_info STATIC
  ${AML_CORE_DIR}/aml_dispatch_info.cc
)
target_include_directories(aml_dispatch_info PUBLIC
  ${AML_COMMON_INCLUDES}
  ${AML_CORE_DIR}
)

# Runtime-only (LiteRtDispatch_Aml). Must NOT link libcompiler_adla.so.
add_library(aml_runtime STATIC
  ${AML_CORE_DIR}/nnsdk_func.cc
  ${AML_CORE_DIR}/adla_log.cc
)
target_include_directories(aml_runtime PUBLIC ${AML_COMMON_INCLUDES})
target_link_libraries(aml_runtime
  PUBLIC
    aml_dispatch_info
    tensorflow-lite
    litert_runtime_c_api_static
    litert_cc_api
    litert_cc_options
    litert_c_options
    litert_compiler_plugin
    absl::strings
    absl::status
    absl::statusor
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(aml_runtime PUBLIC dl)
endif()

# Compiler core is prebuilt (aml_compiler_core / aml_compiler_core_lib /
# aml_compiler_core_so aliases from AmlCompilerCorePrebuilt.cmake).
# LiteRT links that .so; it does not compile compiler_core sources and does
# not link libcompiler_adla.so.

# Backward-compatible name: historical aml_core mixed runtime+compiler.
# Prefer aml_runtime / aml_compiler_core_lib for new targets.
add_library(aml_core INTERFACE)
target_link_libraries(aml_core INTERFACE
  aml_runtime
  aml_compiler_core_lib
  aml_dispatch_info
)

add_library(aml_dispatch_so SHARED
  ${AML_DISPATCH_DIR}/dispatch_api.cc
  ${AML_DISPATCH_DIR}/litert_dispatch_device_context.cc
  ${AML_DISPATCH_DIR}/litert_dispatch_invocation_context.cc
)

target_include_directories(aml_dispatch_so
  PUBLIC
    ${AML_COMMON_INCLUDES}
    ${AML_DISPATCH_DIR}
)

target_link_libraries(aml_dispatch_so
  PUBLIC
    aml_runtime
    litert_runtime_c_api_static
    absl::log
    absl::str_format
    absl::strings
    absl::span
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(aml_dispatch_so PRIVATE dl)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(aml_dispatch_so PRIVATE android log)
endif()

set_target_properties(aml_dispatch_so PROPERTIES
  OUTPUT_NAME "LiteRtDispatch_Aml"
  POSITION_INDEPENDENT_CODE ON
)
if(UNIX AND NOT APPLE)
  target_link_options(aml_dispatch_so PRIVATE
    "-Wl,-soname=libLiteRtDispatch_Aml.so")
  target_link_libraries(aml_dispatch_so PRIVATE
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

add_library(aml_compiler_plugin_so SHARED
  ${AML_COMPILER_DIR}/aml_compiler_plugin.cc
  ${AML_COMPILER_DIR}/node_binary_info.cc
)

target_include_directories(aml_compiler_plugin_so
  PUBLIC
    ${AML_COMMON_INCLUDES}
    ${AML_COMPILER_DIR}
  PRIVATE
    ${TFLITE_BUILD_DIR}/flatbuffers/include
)

# RTTI required for NodeBuilder::SetParameter(dynamic_cast) against
# objects created inside prebuilt libcompiler_adla.so.
target_compile_options(aml_compiler_plugin_so PRIVATE -frtti)
# Keep default visibility so typeinfo can merge across DSO with ADLA.
set_target_properties(aml_compiler_plugin_so PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

target_link_libraries(aml_compiler_plugin_so
  PUBLIC
    aml_compiler_core_lib
    aml_dispatch_info
    litert_compiler_plugin
    absl::flat_hash_map
    absl::str_format
    absl::strings
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(aml_compiler_plugin_so PRIVATE dl)
endif()

set_target_properties(aml_compiler_plugin_so PROPERTIES
  OUTPUT_NAME "LiteRtCompilerPlugin_Aml"
  POSITION_INDEPENDENT_CODE ON
)
if(UNIX AND NOT APPLE)
  target_link_options(aml_compiler_plugin_so PRIVATE
    "-Wl,-soname=libLiteRtCompilerPlugin_Aml.so"
    # Export LiteRt* + RTTI; hide std::filesystem etc so they resolve from
    # system C++ runtime (avoids hijacking libcompiler_adla.so Compile()).
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/compiler/export_aml_compiler_plugin_linux.lds"
    "LINKER:-rpath,$ORIGIN"
    "LINKER:--disable-new-dtags"
    "LINKER:--allow-shlib-undefined"
  )
  target_link_libraries(aml_compiler_plugin_so PRIVATE
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

# Convenience install for board / buildroot rootfs packaging.
install(TARGETS
  litert_runtime_c_api_shared_lib
  aml_dispatch_so
  aml_compiler_plugin_so
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)

install(FILES
  "${AML_COMPILER_CORE_LIB}"
  DESTINATION lib
)
install(FILES
  "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_api.h"
  "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_context.h"
  DESTINATION include/aml_npu/compiler_core
)
if(AML_COMPILER_ADLA_LIB)
  install(FILES "${AML_COMPILER_ADLA_LIB}" DESTINATION lib)
endif()
