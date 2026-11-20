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

# One GitHub fetch: amlnn-toolkit/.../nn_runtime (nnsdk + aml_compiler_core).
# Same idea as Bazel @aml_npu_nn_runtime / LITERT_AML_NN_RUNTIME.
set(_AMLNN_TOOLKIT_COMMIT "5efa2cd45971766bb7cb1f2ddab727e3ce301463")

# Resolve nn_runtime root (contains nnsdk/ + aml_compiler_core/).
if(NOT AML_NN_RUNTIME_ROOT)
  set(_aml_nn_rt_candidates)
  if(DEFINED ENV{LITERT_AML_NN_RUNTIME} AND NOT "$ENV{LITERT_AML_NN_RUNTIME}" STREQUAL "")
    list(APPEND _aml_nn_rt_candidates "$ENV{LITERT_AML_NN_RUNTIME}")
  endif()
  list(APPEND _aml_nn_rt_candidates "${CMAKE_BINARY_DIR}/_deps/aml_nn_runtime")
  # Legacy: separate roots still accepted.
  if(DEFINED ENV{LITERT_AML_COMPILER_CORE} AND NOT "$ENV{LITERT_AML_COMPILER_CORE}" STREQUAL "")
    get_filename_component(_legacy_cc_parent "$ENV{LITERT_AML_COMPILER_CORE}/.." ABSOLUTE)
    list(APPEND _aml_nn_rt_candidates "${_legacy_cc_parent}")
  endif()
  foreach(_cand IN LISTS _aml_nn_rt_candidates)
    if(EXISTS "${_cand}/aml_compiler_core/include/compiler_core_context.h" AND
       EXISTS "${_cand}/nnsdk/include/nnsdk2.h")
      set(AML_NN_RUNTIME_ROOT "${_cand}")
      break()
    endif()
  endforeach()
endif()

if(NOT AML_NN_RUNTIME_ROOT)
  set(_nn_archive "${CMAKE_BINARY_DIR}/_deps/amlnn-toolkit-${_AMLNN_TOOLKIT_COMMIT}.tar.gz")
  set(_nn_extract "${CMAKE_BINARY_DIR}/_deps/amlnn-toolkit-src")
  set(_nn_url "https://github.com/Amlogic-NN/amlnn-toolkit/archive/${_AMLNN_TOOLKIT_COMMIT}.tar.gz")
  set(_nn_rt_in_archive
    "${_nn_extract}/amlnn-toolkit-${_AMLNN_TOOLKIT_COMMIT}/amlnn_runtime/nn_runtime")
  if(NOT EXISTS "${_nn_rt_in_archive}/aml_compiler_core/include/compiler_core_context.h")
    if(NOT EXISTS "${_nn_archive}")
      message(STATUS "Downloading AML amlnn-toolkit archive (nn_runtime)...")
      file(DOWNLOAD "${_nn_url}" "${_nn_archive}" SHOW_PROGRESS STATUS _nn_dl_status)
      list(GET _nn_dl_status 0 _nn_dl_code)
      if(NOT _nn_dl_code EQUAL 0)
        message(FATAL_ERROR "AML nn_runtime download failed (${_nn_dl_status})")
      endif()
    endif()
    if(EXISTS "${_nn_archive}")
      file(MAKE_DIRECTORY "${_nn_extract}")
      file(ARCHIVE_EXTRACT INPUT "${_nn_archive}" DESTINATION "${_nn_extract}")
    endif()
  endif()
  if(EXISTS "${_nn_rt_in_archive}/aml_compiler_core/include/compiler_core_context.h")
    set(_nn_stable "${CMAKE_BINARY_DIR}/_deps/aml_nn_runtime")
    file(MAKE_DIRECTORY "${_nn_stable}")
    file(COPY "${_nn_rt_in_archive}/" DESTINATION "${_nn_stable}")
    set(AML_NN_RUNTIME_ROOT "${_nn_stable}")
  endif()
endif()

if(NOT AML_NN_RUNTIME_ROOT)
  message(FATAL_ERROR
    "AML nn_runtime not found. Set LITERT_AML_NN_RUNTIME to a tree with "
    "nnsdk/ + aml_compiler_core/, or allow GitHub download.")
endif()
set(AML_NN_RUNTIME_ROOT "${AML_NN_RUNTIME_ROOT}" CACHE PATH
  "nn_runtime root (nnsdk/ + aml_compiler_core/)" FORCE)

# compiler_core
if(NOT AML_COMPILER_CORE_ROOT)
  if(DEFINED ENV{LITERT_AML_COMPILER_CORE} AND NOT "$ENV{LITERT_AML_COMPILER_CORE}" STREQUAL "")
    set(AML_COMPILER_CORE_ROOT "$ENV{LITERT_AML_COMPILER_CORE}")
  else()
    set(AML_COMPILER_CORE_ROOT "${AML_NN_RUNTIME_ROOT}/aml_compiler_core")
  endif()
endif()
set(AML_COMPILER_CORE_ROOT "${AML_COMPILER_CORE_ROOT}" CACHE PATH
  "aml_compiler_core root (include/ + libs/)" FORCE)
include("${AML_REPO_ROOT}/third_party/aml_nn_runtime/AmlCompilerCorePrebuilt.cmake")

# nnsdk headers
if(NOT AML_NNSDK_INCLUDE_DIRS)
  set(_aml_nnsdk_candidates)
  if(DEFINED ENV{LITERT_AML_SDK} AND NOT "$ENV{LITERT_AML_SDK}" STREQUAL "")
    list(APPEND _aml_nnsdk_candidates
      "$ENV{LITERT_AML_SDK}/include"
      "$ENV{LITERT_AML_SDK}"
    )
  endif()
  list(APPEND _aml_nnsdk_candidates
    "${AML_NN_RUNTIME_ROOT}/nnsdk/include"
  )
  foreach(_cand IN LISTS _aml_nnsdk_candidates)
    if(EXISTS "${_cand}/nnsdk2.h")
      set(AML_NNSDK_INCLUDE_DIRS "${_cand}")
      set(AML_NNSDK_INCLUDE_DIR "${_cand}")
      message(STATUS "AML nnsdk headers at: ${_cand}")
      break()
    endif()
  endforeach()
endif()
if(NOT AML_NNSDK_INCLUDE_DIRS)
  message(FATAL_ERROR "nnsdk headers not found under ${AML_NN_RUNTIME_ROOT}/nnsdk/include")
endif()

# Optional: Linux packaging dir (compiler_adla / nnsdk .so next to binaries).
# Libs come from the fetched nn_runtime.
if(NOT AML_LINUX_LIB_DIR)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    if(DEFINED ENV{OECORE_TARGET_SYSROOT} OR DEFINED ENV{OE_CMAKE_TOOLCHAIN_FILE})
      set(_aml_pack_dir "${AML_NN_RUNTIME_ROOT}/aml_compiler_core/libs/yocto/aarch64")
    else()
      set(_aml_pack_dir "${AML_NN_RUNTIME_ROOT}/aml_compiler_core/libs/linux/aarch64")
    endif()
  else()
    set(_aml_pack_dir "${AML_NN_RUNTIME_ROOT}/aml_compiler_core/libs/linux/x86_64")
  endif()
  if(EXISTS "${_aml_pack_dir}")
    set(AML_NPU_RUNTIME_LIB_DIR "${_aml_pack_dir}")
  endif()
endif()
if(AML_NPU_RUNTIME_LIB_DIR)
  set(AML_LINUX_LIB_DIR "${AML_NPU_RUNTIME_LIB_DIR}" CACHE PATH
    "Directory with Linux ARM prebuilt libs (libcompiler_adla.so, libnnsdk.so)")
elseif(AML_LINUX_LIB_DIR)
  set(AML_NPU_RUNTIME_LIB_DIR "${AML_LINUX_LIB_DIR}")
else()
  message(WARNING
    "No AML_LINUX_LIB_DIR; set it if binaries need libcompiler_adla.so / libnnsdk.so beside them")
endif()

# TFLite FlatBuffers v25 must precede vendor FlatBuffers if any leftover path leaks.
if(NOT AML_COMPILER_CORE_INCLUDE_DIRS)
  set(AML_COMPILER_CORE_INCLUDE_DIRS "${AML_COMPILER_CORE_INCLUDE_DIR}")
endif()
set(AML_COMMON_INCLUDES
  ${TFLITE_BUILD_DIR}/flatbuffers/include
  ${AML_COMPILER_CORE_INCLUDE_DIRS}
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
    "-Wl,--exclude-libs,libc++_static.a"
    "-Wl,--exclude-libs,libc++abi.a"
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
# Generic ATS (//litert/ats:ats) is built from litert/ats/CMakeLists.txt when
# LITERT_ENABLE_AML=ON. Board wrapper: run_aml_ats.sh (installed with ats).
install(TARGETS
  litert_runtime_c_api_shared_lib
  aml_dispatch_so
  aml_compiler_plugin_so
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats.sh")
  install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats.sh" DESTINATION bin)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats_collect_issues.sh")
  install(PROGRAMS
      "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats_collect_issues.sh"
      DESTINATION bin
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats_gtest.sh")
  install(PROGRAMS
      "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats_gtest.sh"
      DESTINATION bin
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats_extra_models.sh")
  install(PROGRAMS
      "${CMAKE_CURRENT_SOURCE_DIR}/run_aml_ats_extra_models.sh"
      DESTINATION bin
  )
endif()

install(FILES
  "${AML_COMPILER_CORE_LIB}"
  DESTINATION lib
)
# GitHub flat include/ or legacy aml_npu/compiler_core/ nesting.
set(_aml_install_hdr_api "${AML_COMPILER_CORE_INCLUDE_DIR}/compiler_core_api.h")
set(_aml_install_hdr_ctx "${AML_COMPILER_CORE_INCLUDE_DIR}/compiler_core_context.h")
if(NOT EXISTS "${_aml_install_hdr_api}")
  set(_aml_install_hdr_api "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_api.h")
  set(_aml_install_hdr_ctx "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_context.h")
endif()
install(FILES
  "${_aml_install_hdr_api}"
  "${_aml_install_hdr_ctx}"
  DESTINATION include/aml_npu/compiler_core
)
if(AML_COMPILER_ADLA_LIB)
  install(FILES "${AML_COMPILER_ADLA_LIB}" DESTINATION lib)
endif()
