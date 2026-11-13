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

# LiteRT C++ API
#
# LINT.IfChange
cmake_minimum_required(VERSION 3.20)

project(litert_cc_api LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress deprecation warnings (matching Bazel build behavior in .bazelrc).
# The codebase has deprecated Error constructors that are still in use.
# These are GCC/Clang spellings. cl.exe parses "-Wno-..." as its /W<level>
# switch and stops with "D8021: invalid numeric argument
# '/Wno-deprecated-declarations'" on every translation unit, including the
# Abseil and FlatBuffers sources fetched below; clang-cl accepts the -Wno-
# options but warns about -fpermissive. MSVC-style drivers (cl.exe, and
# clang-cl, which CMake also reports as MSVC) get the equivalent /wd switch
# instead. This mirrors litert/CMakeLists.txt.
if(MSVC)
  add_compile_options(
    /wd4996  # deprecated declarations
  )
else()
  add_compile_options(
    -Wno-deprecated-declarations
    -Wno-return-type
    -fpermissive
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/litert/build_common/build_config.h.in")
  set(LITERT_CC_SDK_TREE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../build_common/build_config.h.in")
  set(LITERT_CC_SDK_TREE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
else()
  message(FATAL_ERROR
      "Could not locate the LiteRT source root for cc_sdk from ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(LITERT_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
set(LITERT_BUILD_COMMON_GENERATED_DIR "${LITERT_GENERATED_INCLUDE_DIR}/litert/build_common")
file(MAKE_DIRECTORY "${LITERT_BUILD_COMMON_GENERATED_DIR}")

configure_file(
    ${LITERT_CC_SDK_TREE_ROOT}/litert/build_common/build_config.h.in
    ${LITERT_BUILD_COMMON_GENERATED_DIR}/build_config.h
)
include_directories(${LITERT_GENERATED_INCLUDE_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
if(EXISTS "${LITERT_CC_SDK_TREE_ROOT}/tflite/tools/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LITERT_CC_SDK_TREE_ROOT}/tflite/tools/cmake/modules")
endif()

find_package(absl REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(opencl_headers QUIET)

# C API library
if (WIN32)
    set(LITERT_C_API_LIB_NAME "libLiteRt.dll")
elseif (APPLE)
    set(LITERT_C_API_LIB_NAME "libLiteRt.dylib")
else()
    set(LITERT_C_API_LIB_NAME "libLiteRt.so")
endif()
add_library(litert_c_api SHARED IMPORTED)
set_target_properties(litert_c_api PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/${LITERT_C_API_LIB_NAME}"
)

# Generates an MSVC import library for `dll` at `out_lib` from the DLL's export
# table: dumpbin /exports -> .def -> lib /def. Fails configuration with an
# actionable message when the DLL or MSVC binutils are not available.
function(litert_cc_sdk_generate_implib dll out_lib)
  if(NOT EXISTS "${dll}")
    message(FATAL_ERROR
        "Prebuilt runtime DLL not found at ${dll}. Copy libLiteRt.dll next "
        "to CMakeLists.txt before configuring.")
  endif()

  get_filename_component(_linker_dir "${CMAKE_LINKER}" DIRECTORY)
  find_program(LITERT_DUMPBIN NAMES dumpbin HINTS "${_linker_dir}")
  find_program(LITERT_LIB_TOOL NAMES lib llvm-lib HINTS "${_linker_dir}")
  if(NOT LITERT_DUMPBIN OR NOT LITERT_LIB_TOOL)
    message(FATAL_ERROR
        "Cannot generate an import library for ${dll}: dumpbin and lib (or "
        "llvm-lib) were not found. Run CMake from a Visual Studio developer "
        "shell, or place a matching libLiteRt.lib next to the DLL.")
  endif()

  execute_process(
    COMMAND "${LITERT_DUMPBIN}" /nologo /exports "${dll}"
    OUTPUT_VARIABLE _exports
    RESULT_VARIABLE _result
  )
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "dumpbin /exports failed for ${dll} (exit ${_result}).")
  endif()

  # Export table rows look like "   123   7A 0001F2C0 LiteRtCreateEnvironment".
  string(REGEX MATCHALL "\n +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +[^ \t\r\n]+"
         _rows "${_exports}")
  list(LENGTH _rows _count)
  if(_count EQUAL 0)
    message(FATAL_ERROR "No exports found in ${dll}; cannot generate ${out_lib}.")
  endif()
  get_filename_component(_dll_name "${dll}" NAME)
  set(_def "LIBRARY ${_dll_name}\nEXPORTS\n")
  foreach(_row IN LISTS _rows)
    string(REGEX REPLACE ".* ([^ \t\r\n]+)$" "\\1" _symbol "${_row}")
    string(APPEND _def "${_symbol}\n")
  endforeach()
  set(_def_file "${out_lib}.def")
  file(WRITE "${_def_file}" "${_def}")

  if(CMAKE_C_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
    set(_machine ARM64)
  elseif(CMAKE_C_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    set(_machine X64)
  else()
    message(FATAL_ERROR
        "Unsupported target architecture: ${CMAKE_C_COMPILER_ARCHITECTURE_ID}")
  endif()
  execute_process(
    COMMAND "${LITERT_LIB_TOOL}" /nologo "/def:${_def_file}"
            "/machine:${_machine}" "/out:${out_lib}"
    RESULT_VARIABLE _result
  )
  if(NOT _result EQUAL 0 OR NOT EXISTS "${out_lib}")
    message(FATAL_ERROR "lib /def failed for ${_def_file} (exit ${_result}).")
  endif()
  message(STATUS
      "Generated ${out_lib} (${_count} exports, /machine:${_machine}) from ${_dll_name}")
endfunction()

if(WIN32)
  # On Windows a consumer links against an import library (.lib), not the DLL
  # itself, and CMake refuses to generate ("IMPORTED_IMPLIB not set for imported
  # target \"litert_c_api\"") without one. The prebuilt packages ship only
  # libLiteRt.dll. Use a libLiteRt.lib placed next to the DLL when there is one;
  # otherwise generate it from the DLL's export table into the build directory.
  # Every exported data symbol (e.g. kLiteRtRuntimeBuiltin) is declared
  # __declspec(dllimport) in the public headers, so a .def without DATA
  # annotations links correctly.
  set(LITERT_C_API_DLL "${CMAKE_CURRENT_SOURCE_DIR}/${LITERT_C_API_LIB_NAME}")
  set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${LITERT_C_API_DLL}"
  )
  set(LITERT_C_API_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.lib")
  if(NOT EXISTS "${LITERT_C_API_IMPLIB}")
    set(LITERT_C_API_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/libLiteRt.lib")
    if(NOT EXISTS "${LITERT_C_API_IMPLIB}"
       OR "${LITERT_C_API_DLL}" IS_NEWER_THAN "${LITERT_C_API_IMPLIB}")
      litert_cc_sdk_generate_implib(
        "${LITERT_C_API_DLL}"
        "${LITERT_C_API_IMPLIB}")
    endif()
  endif()
  set_target_properties(litert_c_api PROPERTIES
    IMPORTED_IMPLIB "${LITERT_C_API_IMPLIB}"
  )
endif()

# GPU Accelerator library
if (APPLE)
  set(LITERT_METAL_ACCELERATOR_LIB_NAME "libLiteRtMetalAccelerator.dylib")
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${LITERT_METAL_ACCELERATOR_LIB_NAME}")
    add_library(litert_metal_accelerator SHARED IMPORTED)
    set_target_properties(litert_metal_accelerator PROPERTIES
      IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/${LITERT_METAL_ACCELERATOR_LIB_NAME}"
    )
  endif()
endif()

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/litert_cc_sdk_hdrs.txt FILE_CONTENTS)
string(REGEX REPLACE "[ \t\r\n]+" ";" LITERT_CC_HEADERS "${FILE_CONTENTS}")
list(REMOVE_ITEM LITERT_CC_HEADERS "")
list(TRANSFORM LITERT_CC_HEADERS PREPEND "${LITERT_CC_SDK_TREE_ROOT}/")

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/litert_cc_sdk_srcs.txt FILE_CONTENTS)
string(REGEX REPLACE "[ \t\r\n]+" ";" LITERT_CC_SRCS "${FILE_CONTENTS}")
list(REMOVE_ITEM LITERT_CC_SRCS "")
list(TRANSFORM LITERT_CC_SRCS PREPEND "${LITERT_CC_SDK_TREE_ROOT}/")


# Core C++ API library
add_library(litert_cc_api STATIC
    ${LITERT_CC_SRCS}
)

# Set PUBLIC_HEADER property for proper installation
set_target_properties(litert_cc_api PROPERTIES
    PUBLIC_HEADER "${LITERT_CC_HEADERS}"
)

target_include_directories(litert_cc_api
    PUBLIC
        $<BUILD_INTERFACE:${LITERT_CC_SDK_TREE_ROOT}>
        $<BUILD_INTERFACE:${LITERT_GENERATED_INCLUDE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_cc_api
    PUBLIC
        absl::status
        absl::flat_hash_map
        absl::statusor
        absl::span
        absl::strings
        absl::log
        absl::die_if_null
        flatbuffers::flatbuffers
    PRIVATE
        litert_c_api
)

# run_model_simple target for testing.
add_executable(run_model_simple
    ${LITERT_CC_SDK_TREE_ROOT}/litert/tools/run_model_simple.cc
)

target_include_directories(run_model_simple
    PRIVATE
        ${LITERT_CC_SDK_TREE_ROOT}
)

target_link_libraries(run_model_simple
    PRIVATE
        litert_cc_api
        absl::flags
        absl::flags_parse
        absl::random_random
        absl::log
        absl::log_internal_message
        absl::log_internal_check_op
        absl::string_view
        absl::span
)

# dump_model_simple target for testing.
add_executable(dump_model_simple
    ${LITERT_CC_SDK_TREE_ROOT}/litert/tools/dump_model_simple.cc
)

target_include_directories(dump_model_simple
    PRIVATE
        ${LITERT_CC_SDK_TREE_ROOT}
)

target_link_libraries(dump_model_simple
    PRIVATE
        litert_cc_api
        absl::span
        absl::log
        absl::log_internal_message
        absl::log_internal_check_op
)
# Google-internal LINT.ThenChange()
