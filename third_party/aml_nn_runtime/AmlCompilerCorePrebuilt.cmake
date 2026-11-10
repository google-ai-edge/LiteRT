# Import prebuilt libaml_compiler_core.so for LiteRT / TFLite AML builds.
#
# Expects AML_COMPILER_CORE_ROOT to point at the aml_compiler_core root
# (include/ + libs/), usually ${LITERT_AML_NN_RUNTIME}/aml_compiler_core.
#
# Provides:
#   AML_COMPILER_CORE_INCLUDE_DIR
#   AML_COMPILER_CORE_LIB
#   imported target: aml_compiler_core
#   aliases:         aml_compiler_core_lib, aml_compiler_core_so

if(TARGET aml_compiler_core)
  return()
endif()

if(NOT AML_COMPILER_CORE_ROOT)
  message(FATAL_ERROR "AML_COMPILER_CORE_ROOT is not set")
endif()

set(AML_COMPILER_CORE_INCLUDE_DIR "${AML_COMPILER_CORE_ROOT}/include")

set(_aml_cc_lib "")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  if(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/android/armeabi-v7a/libaml_compiler_core.so")
  else()
    set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/android/arm64-v8a/libaml_compiler_core.so")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    if(DEFINED ENV{OECORE_TARGET_SYSROOT} OR DEFINED ENV{OE_CMAKE_TOOLCHAIN_FILE})
      set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/yocto/aarch64/libaml_compiler_core.so")
    endif()
    if(NOT EXISTS "${_aml_cc_lib}")
      set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/linux/aarch64/libaml_compiler_core.so")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM)")
    if(DEFINED ENV{OECORE_TARGET_SYSROOT} OR DEFINED ENV{OE_CMAKE_TOOLCHAIN_FILE})
      set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/yocto/armv7/libaml_compiler_core.so")
    endif()
    if(NOT EXISTS "${_aml_cc_lib}")
      set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/linux/armv7/libaml_compiler_core.so")
    endif()
  else()
    set(_aml_cc_lib "${AML_COMPILER_CORE_ROOT}/libs/linux/x86_64/libaml_compiler_core.so")
  endif()
endif()

if(NOT _aml_cc_lib OR NOT EXISTS "${_aml_cc_lib}")
  message(FATAL_ERROR
    "Prebuilt libaml_compiler_core.so not found.\n"
    "  looked for: ${_aml_cc_lib}\n"
    "  Set LITERT_AML_NN_RUNTIME (or LITERT_AML_COMPILER_CORE) / place libs under ${AML_COMPILER_CORE_ROOT}/libs/<abi>/")
endif()

# GitHub layout: include/compiler_core_context.h
# Legacy aml_npu layout: include/aml_npu/compiler_core/compiler_core_context.h
set(_aml_cc_hdr "")
if(EXISTS "${AML_COMPILER_CORE_INCLUDE_DIR}/compiler_core_context.h")
  set(_aml_cc_hdr "${AML_COMPILER_CORE_INCLUDE_DIR}/compiler_core_context.h")
elseif(EXISTS "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_context.h")
  set(_aml_cc_hdr "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_context.h")
endif()
if(NOT _aml_cc_hdr)
  message(FATAL_ERROR
    "compiler_core public header missing under:\n"
    "  ${AML_COMPILER_CORE_INCLUDE_DIR}")
endif()

set(AML_COMPILER_CORE_LIB "${_aml_cc_lib}")

# Bazel exposes flat include/*.h as aml_npu/compiler_core/*.h
# (include_prefix + strip_include_prefix). CMake callers use that path.
set(AML_COMPILER_CORE_INCLUDE_DIRS "${AML_COMPILER_CORE_INCLUDE_DIR}")
if(EXISTS "${AML_COMPILER_CORE_INCLUDE_DIR}/compiler_core_context.h"
   AND NOT EXISTS "${AML_COMPILER_CORE_INCLUDE_DIR}/aml_npu/compiler_core/compiler_core_context.h")
  set(_aml_cc_prefixed_root "${CMAKE_BINARY_DIR}/aml_compiler_core_prefixed_include")
  set(_aml_cc_prefixed_parent "${_aml_cc_prefixed_root}/aml_npu")
  set(_aml_cc_prefixed_link "${_aml_cc_prefixed_parent}/compiler_core")
  file(MAKE_DIRECTORY "${_aml_cc_prefixed_parent}")
  if(EXISTS "${_aml_cc_prefixed_link}" OR IS_SYMLINK "${_aml_cc_prefixed_link}")
    file(REMOVE_RECURSE "${_aml_cc_prefixed_link}")
  endif()
  file(CREATE_LINK
    "${AML_COMPILER_CORE_INCLUDE_DIR}"
    "${_aml_cc_prefixed_link}"
    SYMBOLIC)
  list(PREPEND AML_COMPILER_CORE_INCLUDE_DIRS "${_aml_cc_prefixed_root}")
endif()

message(STATUS "AML prebuilt compiler core: ${AML_COMPILER_CORE_LIB}")
message(STATUS "AML compiler core headers: ${AML_COMPILER_CORE_INCLUDE_DIRS}")

add_library(aml_compiler_core SHARED IMPORTED GLOBAL)
set_target_properties(aml_compiler_core PROPERTIES
  IMPORTED_LOCATION "${AML_COMPILER_CORE_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${AML_COMPILER_CORE_INCLUDE_DIRS}"
  IMPORTED_NO_SONAME FALSE
)

add_library(aml_compiler_core_lib ALIAS aml_compiler_core)
add_library(aml_compiler_core_so ALIAS aml_compiler_core)
