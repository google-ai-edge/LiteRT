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

set(LITERT_CC_PUBLIC_HEADERS
    litert_any.h
    litert_buffer_ref.h
    litert_common.h
    litert_compiled_model.h
    litert_custom_op_kernel.h
    litert_element_type.h
    litert_environment_options.h
    litert_environment.h
    litert_event.h
    litert_expected.h
    litert_layout.h
    litert_macros.h
    litert_model.h
    litert_model_types.h
    litert_opaque_options.h
    litert_options.h
    litert_profiler.h
    litert_ranked_tensor_type.h
    litert_tensor_buffer_requirements.h
    litert_tensor_buffer_types.h
    litert_tensor_buffer.h
    options/litert_compiler_options.h
    options/litert_cpu_options.h
    options/litert_google_tensor_options.h
    options/litert_gpu_options.h
    options/litert_intel_openvino_options.h
    options/litert_magic_number_options.h
    options/litert_mediatek_options.h
    options/litert_qualcomm_options.h
    options/litert_runtime_options.h
    options/litert_samsung_options.h
    options/litert_webnn_options.h
)

set(LITERT_CC_REQUIRED_INTERNAL_HEADERS
    internal/litert_c_types_printing.h
    internal/litert_consts.h
    internal/litert_detail.h
    internal/litert_handle.h
    internal/litert_logging.h
    internal/litert_model_predicates.h
    internal/litert_numerics.h
    internal/litert_op_options.h
    internal/litert_platform_support.h
    internal/litert_builder.h
    internal/litert_rng.h
    internal/litert_runtime_builtin.h
    internal/litert_runtime_proxy.h
    internal/litert_shared_library.h
    internal/litert_source_location.h
    internal/litert_tensor_buffer_utils.h
    internal/litert_tflite_error_status_builder.h
    internal/scoped_file.h
    internal/scoped_weight_source.h
)

set(LITERT_CC_PUBLIC_SOURCES
    litert_compiled_model.cc
    litert_custom_op_kernel.cc
    litert_model.cc
    litert_options.cc
    litert_tensor_buffer.cc
)

file(GLOB LITERT_CC_DISCOVERED_PUBLIC_HEADERS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.h"
    "options/*.h"
)
list(SORT LITERT_CC_DISCOVERED_PUBLIC_HEADERS)
set(_litert_cc_expected_public_headers ${LITERT_CC_PUBLIC_HEADERS})
list(SORT _litert_cc_expected_public_headers)
if(NOT LITERT_CC_DISCOVERED_PUBLIC_HEADERS STREQUAL _litert_cc_expected_public_headers)
    message(FATAL_ERROR
        "litert/cc public header manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_CC_DISCOVERED_PUBLIC_HEADERS}\n"
        "Expected:\n  ${_litert_cc_expected_public_headers}\n"
        "Update litert/cc/CMakeLists.txt.")
endif()

file(GLOB LITERT_CC_DISCOVERED_PUBLIC_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.cc"
    "options/*.cc"
)
list(FILTER LITERT_CC_DISCOVERED_PUBLIC_SOURCES EXCLUDE REGEX "_test\\.cc$")
list(SORT LITERT_CC_DISCOVERED_PUBLIC_SOURCES)
set(_litert_cc_expected_public_sources ${LITERT_CC_PUBLIC_SOURCES})
list(SORT _litert_cc_expected_public_sources)
if(NOT LITERT_CC_DISCOVERED_PUBLIC_SOURCES STREQUAL _litert_cc_expected_public_sources)
    message(FATAL_ERROR
        "litert/cc public source manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_CC_DISCOVERED_PUBLIC_SOURCES}\n"
        "Expected:\n  ${_litert_cc_expected_public_sources}\n"
        "Update litert/cc/CMakeLists.txt.")
endif()

set(LITERT_CC_HEADERS
    ${LITERT_CC_REQUIRED_INTERNAL_HEADERS}
    ${LITERT_CC_PUBLIC_HEADERS}
)

# Core C++ API library
# If no public sources, create an empty library to avoid build failures.
# TODO (b/507233667): Convert this to an INTERFACE library.
if(LITERT_CC_PUBLIC_SOURCES)
    add_library(litert_cc_api STATIC
        ${LITERT_CC_PUBLIC_SOURCES}
    )
else()
    add_library(litert_cc_api STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../c/empty.cc
    )
endif()

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
        absl::die_if_null
    PRIVATE
        litert_c_api
        litert_c_options
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
