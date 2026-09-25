# Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

# SDK public headers (compiler_core_*.h, nnsdk2.h / nnsdk.h) are NOT vendored.
# They are consumed as an external library via AML_SDK_ROOT / AML_INCLUDE_DIR.
# Prebuilt .so files also stay in the external AML SDK:
#   ${AML_SDK_ROOT}/android/lib64|lib32/libaml_compiler_core.so
#   ${AML_SDK_ROOT}/linux/buildroot_64|x86/libaml_compiler_core.so
set(AML_SDK_ROOT "" CACHE PATH
  "Root of the external AML SDK (headers + prebuilt .so)")
set(AML_LIB_DIR "" CACHE PATH
  "Directory containing libaml_compiler_core.so")
set(AML_INCLUDE_DIR "" CACHE PATH
  "Directory containing compiler_core_context.h")
set(AML_NNSDK_INCLUDE_DIR "" CACHE PATH
  "Directory containing nnsdk2.h")

if(NOT AML_LIB_DIR AND AML_SDK_ROOT)
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(DEFINED ANDROID_ABI AND ANDROID_ABI STREQUAL "armeabi-v7a")
      set(AML_LIB_DIR "${AML_SDK_ROOT}/android/lib32")
    else()
      set(AML_LIB_DIR "${AML_SDK_ROOT}/android/lib64")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
    set(AML_LIB_DIR "${AML_SDK_ROOT}/linux/x86")
  else()
    set(AML_LIB_DIR "${AML_SDK_ROOT}/linux/buildroot_64")
  endif()
endif()

macro(_aml_locate_header OUT_DIR HEADER)
  if(NOT ${OUT_DIR} AND AML_SDK_ROOT)
    if(EXISTS "${AML_SDK_ROOT}/${HEADER}")
      set(${OUT_DIR} "${AML_SDK_ROOT}")
    elseif(EXISTS "${AML_SDK_ROOT}/include/${HEADER}")
      set(${OUT_DIR} "${AML_SDK_ROOT}/include")
    else()
      file(GLOB_RECURSE _aml_hits "${AML_SDK_ROOT}/*/${HEADER}")
      if(_aml_hits)
        list(GET _aml_hits 0 _aml_first)
        get_filename_component(${OUT_DIR} "${_aml_first}" DIRECTORY)
      endif()
    endif()
  endif()
endmacro()

_aml_locate_header(AML_INCLUDE_DIR compiler_core_context.h)
_aml_locate_header(AML_NNSDK_INCLUDE_DIR nnsdk2.h)

if(NOT AML_INCLUDE_DIR)
  message(WARNING
    "compiler_core_context.h not found. Set -DAML_SDK_ROOT or -DAML_INCLUDE_DIR. "
    "LiteRT does not vendor AML compiler_core headers.")
endif()
if(NOT AML_NNSDK_INCLUDE_DIR)
  message(WARNING
    "nnsdk2.h not found. Set -DAML_SDK_ROOT or -DAML_NNSDK_INCLUDE_DIR. "
    "LiteRT does not vendor NNSDK headers.")
endif()

set(_AML_COMPILER_CORE_SO "")
if(AML_LIB_DIR AND EXISTS "${AML_LIB_DIR}/libaml_compiler_core.so")
  set(_AML_COMPILER_CORE_SO "${AML_LIB_DIR}/libaml_compiler_core.so")
endif()

if(NOT _AML_COMPILER_CORE_SO)
  message(WARNING
    "libaml_compiler_core.so not found. Set -DAML_SDK_ROOT or -DAML_LIB_DIR. "
    "LiteRT does not vendor this library.")
endif()

set(_AML_SDK_INCLUDES "")
if(AML_INCLUDE_DIR)
  list(APPEND _AML_SDK_INCLUDES "${AML_INCLUDE_DIR}")
endif()
if(AML_NNSDK_INCLUDE_DIR AND NOT AML_NNSDK_INCLUDE_DIR STREQUAL AML_INCLUDE_DIR)
  list(APPEND _AML_SDK_INCLUDES "${AML_NNSDK_INCLUDE_DIR}")
endif()

if(_AML_COMPILER_CORE_SO)
  add_library(aml_compiler_core_lib SHARED IMPORTED GLOBAL)
  set_target_properties(aml_compiler_core_lib PROPERTIES
    IMPORTED_LOCATION "${_AML_COMPILER_CORE_SO}"
    INTERFACE_INCLUDE_DIRECTORIES "${_AML_SDK_INCLUDES}")
else()
  add_library(aml_compiler_core_lib INTERFACE)
  if(_AML_SDK_INCLUDES)
    target_include_directories(aml_compiler_core_lib INTERFACE ${_AML_SDK_INCLUDES})
  endif()
endif()

set(AML_COMMON_INCLUDES
  ${TFLITE_BUILD_DIR}/flatbuffers/include
  ${AML_CORE_DIR}
  ${_AML_SDK_INCLUDES}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/..
  ${TENSORFLOW_SOURCE_DIR}
  ${TFLITE_SOURCE_DIR}
)

add_library(aml_dispatch_info STATIC
  ${AML_COMPILER_DIR}/aml_dispatch_info.cc
)
target_include_directories(aml_dispatch_info PUBLIC
  ${AML_COMMON_INCLUDES}
  ${AML_COMPILER_DIR}
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

# Install only LiteRT-produced artifacts. AML SDK .so files stay external.
install(TARGETS
  litert_runtime_c_api_shared_lib
  aml_dispatch_so
  aml_compiler_plugin_so
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)
