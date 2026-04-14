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

# LiteRT C API
cmake_minimum_required(VERSION 3.20)

add_subdirectory(internal)

set(LITERT_C_PUBLIC_HEADERS
    litert_any.h
    litert_builder.h
    litert_common.h
    litert_compiled_model.h
    litert_custom_op_kernel.h
    litert_custom_tensor_buffer.h
    litert_environment_options.h
    litert_environment.h
    litert_event_type.h
    litert_event.h
    litert_gl_types.h
    litert_layout.h
    litert_metrics.h
    litert_model_types.h
    litert_model.h
    litert_op_code.h
    litert_op_options.h
    litert_opaque_options.h
    litert_opencl_types.h
    litert_options.h
    litert_platform_support.h
    litert_profiler_event.h
    litert_profiler.h
    litert_tensor_buffer_requirements.h
    litert_tensor_buffer_types.h
    litert_tensor_buffer.h
    litert_webgpu_types.h
    options/litert_compiler_options.h
    options/litert_cpu_options.h
    options/litert_google_tensor_options.h
    options/litert_google_tensor_options_type.h
    options/litert_gpu_options.h
    options/litert_intel_openvino_options.h
    options/litert_mediatek_options.h
    options/litert_qualcomm_options.h
    options/litert_runtime_options.h
    options/litert_samsung_options.h
    options/litert_webnn_options.h
)

set(LITERT_C_REQUIRED_INTERNAL_HEADERS
    internal/litert_logging.h
    internal/litert_runtime_c_api.h
    internal/litert_dispatch_delegate.h
    internal/litert_scheduling_info.h
)

set(LITERT_C_REQUIRED_VENDOR_HEADERS
    litert_compiler_plugin.h
    litert_compiler_plugin_api.h
    litert_dispatch.h
    litert_dispatch_api.h
)

set(LITERT_C_PUBLIC_SOURCES
    litert_common.cc
    litert_compiled_model.cc
    litert_environment.cc
    litert_environment_options.cc
    litert_event.cc
    litert_layout.cc
    litert_metrics.cc
    litert_model.cc
    litert_op_options.cc
    litert_opaque_options.cc
    litert_options.cc
    litert_profiler.cc
    litert_builder.cc
    litert_tensor_buffer.cc
    litert_tensor_buffer_requirements.cc
    options/litert_compiler_options.cc
    options/litert_cpu_options.cc
    options/litert_google_tensor_options.cc
    options/litert_gpu_options.cc
    options/litert_intel_openvino_options.cc
    options/litert_mediatek_options.cc
    options/litert_qualcomm_options.cc
    options/litert_runtime_options.cc
    options/litert_samsung_options.cc
    options/litert_webnn_options.cc
)

file(GLOB LITERT_C_DISCOVERED_PUBLIC_HEADERS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.h"
    "options/*.h"
)
list(SORT LITERT_C_DISCOVERED_PUBLIC_HEADERS)
set(_litert_c_expected_public_headers ${LITERT_C_PUBLIC_HEADERS})
list(SORT _litert_c_expected_public_headers)
if(NOT LITERT_C_DISCOVERED_PUBLIC_HEADERS STREQUAL _litert_c_expected_public_headers)
    message(FATAL_ERROR
        "litert/c public header manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_C_DISCOVERED_PUBLIC_HEADERS}\n"
        "Expected:\n  ${_litert_c_expected_public_headers}\n"
        "Update litert/c/CMakeLists.txt.")
endif()

file(GLOB LITERT_C_DISCOVERED_PUBLIC_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.cc"
    "options/*.cc"
)
list(FILTER LITERT_C_DISCOVERED_PUBLIC_SOURCES EXCLUDE REGEX "_test\\.cc$")
list(FILTER LITERT_C_DISCOVERED_PUBLIC_SOURCES EXCLUDE REGEX "^empty\\.cc$")
list(SORT LITERT_C_DISCOVERED_PUBLIC_SOURCES)
set(_litert_c_expected_public_sources ${LITERT_C_PUBLIC_SOURCES})
list(SORT _litert_c_expected_public_sources)
if(NOT LITERT_C_DISCOVERED_PUBLIC_SOURCES STREQUAL _litert_c_expected_public_sources)
    message(FATAL_ERROR
        "litert/c public source manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_C_DISCOVERED_PUBLIC_SOURCES}\n"
        "Expected:\n  ${_litert_c_expected_public_sources}\n"
        "Update litert/c/CMakeLists.txt.")
endif()

set(LITERT_C_API_SOURCES
    ${LITERT_C_INTERNAL_SOURCES}
    ${LITERT_C_PUBLIC_SOURCES}
)

set(LITERT_C_HEADERS
    ${LITERT_C_PUBLIC_HEADERS}
    ${LITERT_C_REQUIRED_INTERNAL_HEADERS}
)

# Logging library (shared between core and C API)
add_library(litert_logging
    STATIC
        internal/litert_logging.cc
        internal/litert_logging_helper.cc
        internal/litert_logging_helper_with_runtime_context.cc
)

target_include_directories(litert_logging
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_logging
    PUBLIC
        tensorflow-lite
        absl::strings
        absl::string_view
)

# Platform-specific dependencies
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(litert_logging PUBLIC ${LOG_LIB})
endif()

# Core C API library (static)
add_library(litert_c_api STATIC ${LITERT_C_API_SOURCES})

target_include_directories(litert_c_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/opencl_headers>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_c_api
    PUBLIC
        litert_cc_internal
        litert_core
        litert_core_model
        litert_logging
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        flatbuffers::flatbuffers
        tensorflow-lite
    PRIVATE
        Threads::Threads
)

# Platform-specific dependencies
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(litert_c_api PRIVATE ${LOG_LIB})
endif()

set_target_properties(litert_c_api PROPERTIES
    PUBLIC_HEADER "${LITERT_C_HEADERS}"
)

foreach(header ${LITERT_C_PUBLIC_HEADERS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${header})
        configure_file(${header} ${CMAKE_CURRENT_BINARY_DIR}/include/litert/c/${header} COPYONLY)
    endif()
endforeach()

foreach(header ${LITERT_C_REQUIRED_INTERNAL_HEADERS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${header})
        configure_file(${header} ${CMAKE_CURRENT_BINARY_DIR}/include/litert/c/${header} COPYONLY)
    endif()
endforeach()

foreach(header ${LITERT_C_REQUIRED_VENDOR_HEADERS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../vendors/c/${header})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../vendors/c/${header}
                       ${CMAKE_CURRENT_BINARY_DIR}/include/litert/vendors/c/${header}
                       COPYONLY)
    endif()
endforeach()

# Options subdirectory
add_subdirectory(options)

# C API shared library
add_library(litert_runtime_c_api_shared_lib SHARED empty.cc)
set_target_properties(litert_runtime_c_api_shared_lib PROPERTIES
    OUTPUT_NAME "LiteRt"
    POSITION_INDEPENDENT_CODE ON
)
set_target_properties(litert_runtime_c_api_shared_lib PROPERTIES INTERFACE_LINK_LIBRARIES "")
target_compile_options(litert_runtime_c_api_shared_lib PRIVATE
    -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables
    -ffunction-sections -fdata-sections
)
target_include_directories(litert_runtime_c_api_shared_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/opencl_headers>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set(_litert_runtime_c_api_public_deps
    absl::status
    absl::statusor
    absl::strings
    absl::log
    absl::log_flags
    absl::log_initialize
    absl::log_internal_check_op
    absl::log_internal_format
    absl::log_internal_globals
    absl::log_internal_message
    absl::log_sink
    absl::span
    absl::flat_hash_map
    flatbuffers::flatbuffers
    litert_core
    litert_core_model
    litert_cc_api
    litert_cc_options
)

set(_litert_runtime_c_api_private_archives
    litert_c_api
    litert_runtime
    # Public C++ headers inline the default builtin-runtime path, for example,
    # it references kLiteRtRuntimeBuiltin from litert_cc_internal. Keep that
    # archive in the shared runtime so downstream users linking libLiteRt.so
    # can resolve the symbol without also linking litert_cc_internal.a.
    litert_cc_internal
    tensorflow-lite
    Threads::Threads
)

# Provide missing TensorFlow Lite GPU GL core objects for Android builds.
if(LITERT_PLATFORM_ANDROID)
  add_library(litert_tflite_gpu_gl_core STATIC
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/egl_context.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/egl_environment.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/egl_surface.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_buffer.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_errors.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_program.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_shader.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_texture.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_texture_helper.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/gl_sync.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/android_sync.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/async_buffers.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/object_manager.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/gl/request_gpu_info.cc)
  target_sources(litert_tflite_gpu_gl_core PRIVATE
    ${TFLITE_SOURCE_DIR}/delegates/gpu/cl/egl_sync.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/cl/gl_interop.cc)
  set_source_files_properties(
    ${TFLITE_SOURCE_DIR}/delegates/gpu/cl/egl_sync.cc
    PROPERTIES COMPILE_DEFINITIONS "EGL_EGLEXT_PROTOTYPES")
  target_sources(litert_tflite_gpu_gl_core PRIVATE
    ${TFLITE_SOURCE_DIR}/delegates/utils/async_type_helpers.cc
    ${TFLITE_SOURCE_DIR}/delegates/utils/utils.cc
    ${TFLITE_SOURCE_DIR}/delegates/utils/sync_fence.cc)
  target_sources(litert_tflite_gpu_gl_core PRIVATE
    ${TFLITE_SOURCE_DIR}/delegates/gpu/android_hardware_buffer.cc
    ${TFLITE_SOURCE_DIR}/async/backend_async_kernel_interface.cc)
  set_target_properties(litert_tflite_gpu_gl_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(litert_tflite_gpu_gl_core
    PUBLIC
      $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>)
  target_link_libraries(litert_tflite_gpu_gl_core
    PUBLIC
      absl::status
      absl::strings
      absl::span
      absl::flat_hash_map
      flatbuffers::flatbuffers)
  list(APPEND _litert_runtime_c_api_private_archives litert_tflite_gpu_gl_core)
endif()

# Interface target that exposes static linking of the runtime C API and its
# transitive archives.
add_library(litert_runtime_c_api_static INTERFACE)
target_link_libraries(litert_runtime_c_api_static
    INTERFACE
        ${_litert_runtime_c_api_public_deps}
        ${_litert_runtime_c_api_private_archives}
)

# Propagate platform GL/Metal requirements to consumers when statically linking
if(APPLE)
  target_link_libraries(litert_runtime_c_api_static
      INTERFACE "-framework Metal" "-framework Foundation" "-framework OpenGL")
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_package(OpenGL)
  if (OpenGL_FOUND)
    target_link_libraries(litert_runtime_c_api_static INTERFACE OpenGL::GL)
  endif()
endif()

if(LITERT_PLATFORM_ANDROID)
  find_library(ANDROID_EGL_LIB EGL)
  find_library(ANDROID_GLESV3_LIB GLESv3)
  find_library(ANDROID_GLESV2_LIB GLESv2)
  target_link_libraries(litert_runtime_c_api_static INTERFACE ${LOG_LIB} android log nativewindow ${ANDROID_EGL_LIB} ${ANDROID_GLESV3_LIB} ${ANDROID_GLESV2_LIB})
endif()

target_link_libraries(litert_runtime_c_api_shared_lib
    PUBLIC
        ${_litert_runtime_c_api_public_deps}
)

if(APPLE)
  target_link_libraries(litert_runtime_c_api_shared_lib
      PRIVATE
        ${_litert_runtime_c_api_private_archives})
  target_link_options(litert_runtime_c_api_shared_lib PRIVATE -Wl,-all_load)
else()
  target_link_libraries(litert_runtime_c_api_shared_lib
      PRIVATE
        -Wl,--whole-archive
        ${_litert_runtime_c_api_private_archives}
        -Wl,--no-whole-archive)
endif()
# GPU/GL dependencies by platform
if(APPLE)
  target_link_libraries(litert_runtime_c_api_shared_lib PRIVATE "-framework OpenGL")
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_package(OpenGL)
  if (OpenGL_FOUND)
    target_link_libraries(litert_runtime_c_api_shared_lib PRIVATE OpenGL::GL)
  endif()
endif()
if(LITERT_PLATFORM_ANDROID)
    find_library(ANDROID_EGL_LIB EGL)
    find_library(ANDROID_GLESV3_LIB GLESv3)
    find_library(ANDROID_GLESV2_LIB GLESv2)
    target_link_libraries(litert_runtime_c_api_shared_lib PRIVATE ${LOG_LIB} android log nativewindow ${ANDROID_EGL_LIB} ${ANDROID_GLESV3_LIB} ${ANDROID_GLESV2_LIB})
endif()
