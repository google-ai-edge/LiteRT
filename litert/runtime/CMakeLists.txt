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
    compiled_model.cc
    custom_op_dispatcher.cc
    dispatch/dispatch_delegate.cc
    dispatch/dispatch_delegate_kernel.cc
    dispatch/dispatch_opaque_options.cc
    dispatch/litert_dispatch.cc
    event.cc
    external_litert_buffer_context.cc
    gl_buffer.cc
    gl_texture.cc
    gpu_environment.cc
    open_cl_memory.cc
    open_cl_sync.cc
    profiler.cc
    tensor_buffer.cc
    tensor_buffer_conversion.cc
    tensor_buffer_requirements.cc
    tfl_utils.cc
    webgpu_buffer.cc
)

# Add platform-specific sources
if(LITERT_PLATFORM_ANDROID)
    list(APPEND LITERT_RUNTIME_SOURCES
        ahwb_buffer.cc
        ion_buffer.cc
        fastrpc_buffer.cc
    )
elseif(LITERT_PLATFORM_LINUX)
    list(APPEND LITERT_RUNTIME_SOURCES
        dmabuf_buffer.cc
    )
endif()

# Runtime library
add_library(litert_runtime STATIC ${LITERT_RUNTIME_SOURCES})

target_include_directories(litert_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_runtime
    PUBLIC
        litert_core
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
    target_link_libraries(litert_runtime PRIVATE ${LOG_LIB})
endif()

# Add platform-specific compile definitions
target_compile_definitions(litert_runtime
    PRIVATE
        $<$<BOOL:${LITERT_PLATFORM_MACOS}>:LITERT_PLATFORM_MACOS>
        $<$<BOOL:${LITERT_PLATFORM_LINUX}>:LITERT_PLATFORM_LINUX>
        $<$<BOOL:${LITERT_PLATFORM_ANDROID}>:LITERT_PLATFORM_ANDROID>
        $<$<BOOL:${LITERT_PLATFORM_WINDOWS}>:LITERT_PLATFORM_WINDOWS>
)

# GPU/NPU support
if(NOT LITERT_USE_GPU)
    target_compile_definitions(litert_runtime PRIVATE LITERT_NO_GPU)
endif()
if(NOT LITERT_USE_NPU)
    target_compile_definitions(litert_runtime PRIVATE LITERT_NO_NPU)
endif()