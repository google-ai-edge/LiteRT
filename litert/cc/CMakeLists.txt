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
cmake_minimum_required(VERSION 3.20)

# Core C++ API library
add_library(litert_cc_api STATIC
    litert_compiled_model.cc
    litert_macros.cc
    litert_model.cc
    litert_op_options.cc
    litert_opaque_options.cc
    litert_shared_library.cc
    litert_tensor_buffer.cc
    litert_tensor_buffer_utils.cc
)

target_include_directories(litert_cc_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_cc_api
    PUBLIC
        absl::status
        absl::statusor
        absl::span
        absl::strings
        absl::log
    PRIVATE
        litert_c_api
)

# Export headers
set(LITERT_CC_HEADERS
    litert_element_type.h
    litert_expected.h
    litert_layout.h
    litert_macros.h
    litert_model.h
    litert_options.h
    litert_tensor_buffer.h
    litert_tensor_buffer_requirements.h
    litert_detail.h
    litert_handle.h
    litert_compiled_model.h
    litert_c_types_printing.h
    litert_buffer_ref.h
    litert_any.h
    litert_consts.h
    litert_custom_op_kernel.h
    litert_dispatch_delegate.h
    litert_environment.h
    litert_environment_options.h
    litert_event.h
    litert_logging.h
    litert_model_predicates.h
    litert_numerics.h
    litert_op_options.h
    litert_opaque_options.h
    litert_platform_support.h
    litert_profiler.h
    litert_rng.h
    litert_shared_library.h
    litert_source_location.h
    litert_tensor_buffer_utils.h
    litert_tflite_error_status_builder.h
)

# Set PUBLIC_HEADER property for proper installation
set_target_properties(litert_cc_api PROPERTIES
    PUBLIC_HEADER "${LITERT_CC_HEADERS}"
)

# Also copy headers to build directory for local testing convenience
foreach(header ${LITERT_CC_HEADERS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${header})
        configure_file(${header} ${CMAKE_CURRENT_BINARY_DIR}/include/litert/cc/${header} COPYONLY)
    endif()
endforeach()

# Add subdirectories
add_subdirectory(internal)
add_subdirectory(options)
