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

# Base source files
set(LITERT_RUNTIME_SOURCES
    accelerator_registry.cc
    accelerators/auto_registration.cc
    accelerators/dispatch/dispatch_accelerator.cc
    accelerators/xnnpack/xnnpack_accelerator.cc
    ahwb_buffer.cc
    ahwb_wrapper.cc
    compiled_model.cc
    custom_buffer.cc
    custom_op_dispatcher.cc
    dispatch/dispatch_delegate.cc
    dispatch/dispatch_delegate_kernel.cc
    dispatch/dispatch_opaque_options.cc
    dispatch/litert_dispatch.cc
    dmabuf_buffer.cc
    event.cc
    external_litert_buffer_context.cc
    fastrpc_buffer.cc
    gpu_environment.cc
    # Always compile GL buffer/texture sources; they contain internal stubs
    # when LITERT_HAS_OPENGL_SUPPORT is 0 so they are safe to include on CPU-only builds.
    gl_buffer.cc
    gl_texture.cc
    ion_buffer.cc
    litert_runtime_options.cc
    magic_number_utils.cc
    profiler.cc
    profiler_summarizer.cc
    tensor_buffer.cc
    tensor_buffer_registry.cc
    tensor_buffer_requirements.cc
    tfl_utils.cc
    ${TFLITE_SOURCE_DIR}/delegates/utils/simple_opaque_delegate.cc
    ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
    ${TFLITE_SOURCE_DIR}/profiling/profile_buffer.cc
    ${TFLITE_SOURCE_DIR}/profiling/time.cc
)

# Runtime library
add_library(litert_runtime STATIC ${LITERT_RUNTIME_SOURCES})

set(LITERT_XNNPACK_INCLUDE_DIR "${TFLITE_BUILD_DIR}/xnnpack/include" CACHE PATH "Path to XNNPACK public headers used by LiteRT")
set(LITERT_PTHREADPOOL_INCLUDE_DIR "${TFLITE_BUILD_DIR}/pthreadpool-source/include" CACHE PATH "Path to pthreadpool headers used by LiteRT")

# Optional GPU-related sources
if(APPLE)
  target_sources(litert_runtime PRIVATE metal_info.cc)
endif()
target_sources(litert_runtime PRIVATE open_cl_memory.cc open_cl_sync.cc)

# Link Apple frameworks for Metal and OpenGL interop
if(APPLE)
  set_source_files_properties(metal_info.cc PROPERTIES COMPILE_FLAGS "-x objective-c++")
  target_link_libraries(litert_runtime PRIVATE "-framework Metal" "-framework Foundation" "-framework OpenGL")
endif()

target_include_directories(litert_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${LITERT_XNNPACK_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${LITERT_PTHREADPOOL_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_BUILD_DIR}/opencl_headers>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_runtime
    PUBLIC
        litert_core
        litert_core_cache
        litert_core_model
        litert_c_api
        litert_c_options
        litert_cc_api
        litert_cc_options
        litert_cc_internal
        litert_compiler_plugin
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        absl::synchronization
        flatbuffers::flatbuffers
        tensorflow-lite
    PRIVATE
        Threads::Threads
)

# Platform-specific configurations
if(LITERT_PLATFORM_ANDROID)
    find_library(ANDROID_EGL_LIB EGL)
    find_library(ANDROID_GLESV3_LIB GLESv3)
    target_link_libraries(litert_runtime PUBLIC ${LOG_LIB} nativewindow ${ANDROID_EGL_LIB} ${ANDROID_GLESV3_LIB})
    # Ensure EGL types and KHR extension prototypes are visible in event.cc
    set_source_files_properties(event.cc PROPERTIES
      COMPILE_DEFINITIONS "EGL_EGLEXT_PROTOTYPES"
      COMPILE_OPTIONS "-includeEGL/egl.h;-includeEGL/eglext.h")
endif()

# Add platform-specific compile definitions
target_compile_definitions(litert_runtime
    PRIVATE
        $<$<BOOL:${LITERT_PLATFORM_MACOS}>:LITERT_PLATFORM_MACOS>
        $<$<BOOL:${LITERT_PLATFORM_LINUX}>:LITERT_PLATFORM_LINUX>
        $<$<BOOL:${LITERT_PLATFORM_ANDROID}>:LITERT_PLATFORM_ANDROID>
        $<$<BOOL:${LITERT_PLATFORM_WINDOWS}>:LITERT_PLATFORM_WINDOWS>
)
