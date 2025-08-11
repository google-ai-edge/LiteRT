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

# Core C API library
add_library(litert_c_api STATIC
    litert_accelerator.cc
    litert_accelerator_registration.cc
    litert_common.cc
    litert_compiled_model.cc
    litert_environment.cc
    litert_environment_options.cc
    litert_event.cc
    litert_layout.cc
    litert_logging.cc
    litert_metrics.cc
    litert_model.cc
    litert_op_options.cc
    litert_opaque_options.cc
    litert_options.cc
    litert_profiler.cc
    litert_tensor_buffer.cc
    litert_tensor_buffer_requirements.cc
)

target_include_directories(litert_c_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_c_api
    PUBLIC
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        flatbuffers::flatbuffers
    PRIVATE
        Threads::Threads
)

# Platform-specific dependencies
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(litert_c_api PRIVATE ${LOG_LIB})
endif()

# Set compile definitions
target_compile_definitions(litert_c_api
    PRIVATE
        $<$<NOT:$<BOOL:${LITERT_USE_GPU}>>:LITERT_NO_GPU>
        $<$<NOT:$<BOOL:${LITERT_USE_NPU}>>:LITERT_NO_NPU>
)

# Export headers
set(LITERT_C_HEADERS
    litert_common.h
    litert_model.h
    litert_options.h
    litert_op_code.h
    litert_tensor_buffer.h
    litert_tensor_buffer_requirements.h
    litert_logging.h
    litert_tensor.h
    litert_compiled_model.h
    litert_compiler_plugin.h
    litert_dispatch.h
    litert_dispatch_delegate.h
)


set_target_properties(litert_c_api PROPERTIES
    PUBLIC_HEADER "${LITERT_C_HEADERS}"
)

# Also copy headers to build directory for local testing convenience
foreach(header ${LITERT_C_HEADERS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${header})
        configure_file(${header} ${CMAKE_CURRENT_BINARY_DIR}/include/litert/c/${header} COPYONLY)
    endif()
endforeach()

# Options subdirectory
add_subdirectory(options)
