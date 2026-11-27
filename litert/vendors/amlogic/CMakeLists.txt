# Copyright 2026 Amlogic, Inc.
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

# One GitHub fetch: amlnn-toolkit/amlnn_runtime/nn_runtime
# (nnsdk + aml_compiler_core). Same pin as Bazel @aml_npu_nn_runtime.
# https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_runtime/nn_runtime
set(_AML_TOOLKIT_COMMIT "11410d28ff079656e13a702b9ed8e8ed7d7f381b")
set(_AML_TOOLKIT_SHA256 "49dd665d1d1e51f7a701b8881f3c9b77356136a3ea15c125a3329e4bfca82b09")

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
  get_filename_component(_aml_npu_nn_rt "${AML_REPO_ROOT}/../../aml_npu" ABSOLUTE)
  # aml_npu is not nn_runtime layout; skip unless it gains matching dirs.
  foreach(_cand IN LISTS _aml_nn_rt_candidates)
    set(_cand_ok FALSE)
    if(EXISTS "${_cand}/aml_compiler_core/include/compiler_core_context.h" AND
       EXISTS "${_cand}/nnsdk/include/nnsdk2.h")
      if(NOT _cand STREQUAL "${CMAKE_BINARY_DIR}/_deps/aml_nn_runtime")
        set(_cand_ok TRUE)
      elseif(EXISTS "${_cand}/.aml_nn_pin")
        file(READ "${_cand}/.aml_nn_pin" _cand_pin)
        string(STRIP "${_cand_pin}" _cand_pin)
        if(_cand_pin STREQUAL "${_AML_TOOLKIT_COMMIT}")
          set(_cand_ok TRUE)
        endif()
      endif()
    endif()
    if(_cand_ok)
      set(AML_NN_RUNTIME_ROOT "${_cand}")
      break()
    endif()
  endforeach()
endif()

if(NOT AML_NN_RUNTIME_ROOT)
  set(_nn_archive "${CMAKE_BINARY_DIR}/_deps/amlnn-toolkit-${_AML_TOOLKIT_COMMIT}.tar.gz")
  set(_nn_extract "${CMAKE_BINARY_DIR}/_deps/amlnn-toolkit-src")
  set(_nn_url "https://github.com/Amlogic-NN/amlnn-toolkit/archive/${_AML_TOOLKIT_COMMIT}.tar.gz")
  set(_nn_rt_in_archive
    "${_nn_extract}/amlnn-toolkit-${_AML_TOOLKIT_COMMIT}/amlnn_runtime/nn_runtime")
  if(NOT EXISTS "${_nn_rt_in_archive}/aml_compiler_core/include/compiler_core_context.h")
    if(NOT EXISTS "${_nn_archive}")
      message(STATUS "Downloading amlnn-toolkit nn_runtime (nnsdk + aml_compiler_core)...")
      if(_AML_TOOLKIT_SHA256)
        file(DOWNLOAD "${_nn_url}" "${_nn_archive}" SHOW_PROGRESS STATUS _nn_dl_status
             EXPECTED_HASH "SHA256=${_AML_TOOLKIT_SHA256}")
      else()
        file(DOWNLOAD "${_nn_url}" "${_nn_archive}" SHOW_PROGRESS STATUS _nn_dl_status)
      endif()
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
  if(EXISTS "${_nn_rt_in_archive}/aml_compiler_core/include/compiler_core_context.h" AND
     EXISTS "${_nn_rt_in_archive}/nnsdk/include/nnsdk2.h")
    set(_nn_stable "${CMAKE_BINARY_DIR}/_deps/aml_nn_runtime")
    file(REMOVE_RECURSE "${_nn_stable}")
    file(MAKE_DIRECTORY "${_nn_stable}")
    file(COPY "${_nn_rt_in_archive}/aml_compiler_core" "${_nn_rt_in_archive}/nnsdk"
         DESTINATION "${_nn_stable}")
    file(WRITE "${_nn_stable}/.aml_nn_pin" "${_AML_TOOLKIT_COMMIT}\n")
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
# Prefer libs from the fetched nn_runtime; fall back to legacy aml_npu.
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
if(NOT AML_NPU_ROOT)
  if(DEFINED ENV{AML_NPU_ROOT} AND NOT "$ENV{AML_NPU_ROOT}" STREQUAL "")
    set(AML_NPU_ROOT "$ENV{AML_NPU_ROOT}")
  else()
    get_filename_component(AML_NPU_ROOT "${AML_REPO_ROOT}/../../aml_npu" ABSOLUTE)
  endif()
endif()
set(AML_NPU_ROOT "${AML_NPU_ROOT}" CACHE PATH
  "Optional legacy aml_npu (extra packaging)" FORCE)
if(NOT AML_NPU_RUNTIME_LIB_DIR AND EXISTS "${AML_NPU_ROOT}/runtime_lib/cmake/AmlNpuRuntime.cmake")
  include("${AML_NPU_ROOT}/runtime_lib/cmake/AmlNpuRuntime.cmake")
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
# Lives in vendors/amlogic/core (same as Bazel //litert/vendors/amlogic/core:amlogic_dispatch_info).
add_library(amlogic_dispatch_info STATIC
  ${AML_CORE_DIR}/amlogic_dispatch_info.cc
)
target_include_directories(amlogic_dispatch_info PUBLIC
  ${AML_COMMON_INCLUDES}
  ${AML_CORE_DIR}
)

# Runtime-only (LiteRtDispatch_Amlogic). Must NOT link libcompiler_adla.so.
add_library(amlogic_runtime STATIC
  ${AML_CORE_DIR}/nnsdk_func.cc
)
target_include_directories(amlogic_runtime PUBLIC ${AML_COMMON_INCLUDES})
target_link_libraries(amlogic_runtime
  PUBLIC
    amlogic_dispatch_info
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
  target_link_libraries(amlogic_runtime PUBLIC dl)
endif()

# Compiler core is prebuilt (aml_compiler_core / aml_compiler_core_lib /
# aml_compiler_core_so aliases from AmlCompilerCorePrebuilt.cmake).
# LiteRT never links that .so: the compiler plugin resolves its entry points
# at run time via dlopen (aml_compiler_core_loader), so processes that never
# compile never map libaml_compiler_core.so / libcompiler_adla.so. The .so
# is still installed below for the dlopen to find it.

# Backward-compatible name: historical amlogic_core mixed runtime+compiler.
# Prefer amlogic_runtime / aml_compiler_core_lib for new targets.
add_library(amlogic_core INTERFACE)
target_link_libraries(amlogic_core INTERFACE
  amlogic_runtime
  aml_compiler_core_lib
  amlogic_dispatch_info
)

add_library(amlogic_dispatch_so SHARED
  ${AML_DISPATCH_DIR}/dispatch_api.cc
  ${AML_DISPATCH_DIR}/litert_dispatch_device_context.cc
  ${AML_DISPATCH_DIR}/litert_dispatch_invocation_context.cc
)

target_include_directories(amlogic_dispatch_so
  PUBLIC
    ${AML_COMMON_INCLUDES}
    ${AML_DISPATCH_DIR}
)

target_link_libraries(amlogic_dispatch_so
  PUBLIC
    amlogic_runtime
    litert_runtime_c_api_static
    absl::log
    absl::str_format
    absl::strings
    absl::span
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(amlogic_dispatch_so PRIVATE dl)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(amlogic_dispatch_so PRIVATE android log)
endif()

set_target_properties(amlogic_dispatch_so PROPERTIES
  OUTPUT_NAME "LiteRtDispatch_Amlogic"
  POSITION_INDEPENDENT_CODE ON
)
if(UNIX AND NOT APPLE)
  target_link_options(amlogic_dispatch_so PRIVATE
    "-Wl,-soname=libLiteRtDispatch_Amlogic.so")
  target_link_libraries(amlogic_dispatch_so PRIVATE
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

add_library(amlogic_compiler_plugin_so SHARED
  ${AML_COMPILER_DIR}/amlogic_compiler_plugin.cc
  ${AML_COMPILER_DIR}/node_binary_info.cc
  ${AML_CORE_DIR}/aml_compiler_core_loader.cc
)

target_include_directories(amlogic_compiler_plugin_so
  PUBLIC
    ${AML_COMMON_INCLUDES}
    ${AML_COMPILER_DIR}
  PRIVATE
    ${TFLITE_BUILD_DIR}/flatbuffers/include
)

# Pure C plugin boundary. Do not compile or export adla C++ RTTI; cross-DSO
# dynamic_cast belongs in the ADLA SDK, not this .so.
target_compile_options(amlogic_compiler_plugin_so PRIVATE -fno-rtti)

target_link_libraries(amlogic_compiler_plugin_so
  PUBLIC
    amlogic_dispatch_info
    litert_compiler_plugin
    absl::flat_hash_map
    absl::str_format
    absl::strings
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(amlogic_compiler_plugin_so PRIVATE dl)
endif()

set_target_properties(amlogic_compiler_plugin_so PROPERTIES
  OUTPUT_NAME "LiteRtCompilerPlugin_Amlogic"
  POSITION_INDEPENDENT_CODE ON
)
if(UNIX AND NOT APPLE)
  target_link_options(amlogic_compiler_plugin_so PRIVATE
    "-Wl,-soname=libLiteRtCompilerPlugin_Amlogic.so"
    # Export the LiteRT C plugin ABI only.
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../../build_common/export_litert_only_linux.lds"
    "-Wl,--exclude-libs,libc++_static.a"
    "-Wl,--exclude-libs,libc++abi.a"
    "LINKER:-rpath,$ORIGIN"
    "LINKER:--disable-new-dtags"
    "LINKER:--allow-shlib-undefined"
  )
  target_link_libraries(amlogic_compiler_plugin_so PRIVATE
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
  amlogic_dispatch_so
  amlogic_compiler_plugin_so
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)

install(FILES
  "${AML_COMPILER_CORE_LIB}"
  DESTINATION lib
)
if(AML_COMPILER_ADLA_LIB)
  install(FILES "${AML_COMPILER_ADLA_LIB}" DESTINATION lib)
endif()

if(TARGET GTest::gtest_main)
  add_executable(dispatch_api_amlogic_test
    ${AML_DISPATCH_DIR}/dispatch_api_amlogic_test.cc
  )
  target_link_libraries(dispatch_api_amlogic_test PRIVATE
    amlogic_dispatch_so
    litert_compiler_plugin
    GTest::gtest_main
  )
  if(UNIX AND NOT APPLE)
    target_link_options(dispatch_api_amlogic_test PRIVATE
      "LINKER:-rpath,$ORIGIN")
  endif()

  add_executable(amlogic_compiler_plugin_test
    ${AML_COMPILER_DIR}/compiler_plugin_test.cc
  )
  target_link_libraries(amlogic_compiler_plugin_test PRIVATE
    amlogic_compiler_plugin_so
    litert_compiler_plugin
    GTest::gtest_main
  )
  if(UNIX AND NOT APPLE)
    target_link_options(amlogic_compiler_plugin_test PRIVATE
      "LINKER:-rpath,$ORIGIN")
  endif()
endif()
