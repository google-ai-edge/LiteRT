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

project(litert_cc_api LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LITERT_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
set(LITERT_BUILD_COMMON_GENERATED_DIR "${LITERT_GENERATED_INCLUDE_DIR}/litert/build_common")
file(MAKE_DIRECTORY "${LITERT_BUILD_COMMON_GENERATED_DIR}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../build_common/build_config.h.in
    ${LITERT_BUILD_COMMON_GENERATED_DIR}/build_config.h
)
include_directories(${LITERT_GENERATED_INCLUDE_DIR})

set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tflite" CACHE PATH "Path to TFLite source directory")
list(APPEND CMAKE_MODULE_PATH "${TFLITE_SOURCE_DIR}/tools/cmake/modules")

find_package(absl REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(opencl_headers REQUIRED)

# C API library
add_library(litert_c_api SHARED IMPORTED)
set_target_properties(litert_c_api PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.so"
)

set(LITERT_CC_HEADERS
    ../cc/internal/litert_c_types_printing.h
    ../cc/internal/litert_consts.h
    ../cc/internal/litert_detail.h
    ../cc/internal/litert_handle.h
    ../cc/internal/litert_logging.h
    ../cc/internal/litert_numerics.h
    ../cc/internal/litert_platform_support.h
    ../cc/internal/litert_source_location.h
    ../cc/internal/litert_tensor_buffer_utils.h
    ../cc/internal/litert_tflite_error_status_builder.h
    ../cc/litert_any.h
    ../cc/litert_buffer_ref.h
    ../cc/litert_compiled_model.h
    ../cc/litert_custom_op_kernel.h
    ../cc/litert_dispatch_delegate.h
    ../cc/litert_element_type.h
    ../cc/litert_environment_options.h
    ../cc/litert_environment.h
    ../cc/litert_event.h
    ../cc/litert_expected.h
    ../cc/litert_layout.h
    ../cc/litert_macros.h
    ../cc/litert_model.h
    ../cc/litert_opaque_options.h
    ../cc/litert_options.h
    ../cc/litert_profiler.h
    ../cc/litert_tensor_buffer_requirements.h
    ../cc/litert_tensor_buffer.h
)

set(LITERT_CC_SRCS
    ../c/internal/litert_logging.cc
    ../cc/internal/litert_tensor_buffer_utils.cc
    ../cc/litert_compiled_model.cc
    ../cc/litert_macros.cc
    ../cc/litert_model.cc
    ../cc/litert_opaque_options.cc
    ../cc/litert_options.cc
    ../cc/litert_tensor_buffer.cc
    ../cc/options/litert_compiler_options.cc
    ../cc/options/litert_cpu_options.cc
    ../cc/options/litert_darwinn_options.cc
    ../cc/options/litert_google_tensor_options.cc
    ../cc/options/litert_gpu_options.cc
    ../cc/options/litert_intel_openvino_options.cc
    ../cc/options/litert_magic_number_options.cc
    ../cc/options/litert_mediatek_options.cc
    ../cc/options/litert_qualcomm_options.cc
    ../cc/options/litert_runtime_options.cc
)

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
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
        flatbuffers::flatbuffers
    PRIVATE
        litert_c_api
)

# run_model_simple target for testing.
add_executable(run_model_simple
    ../tools/run_model_simple.cc
)

target_include_directories(run_model_simple
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

target_link_libraries(run_model_simple
    PRIVATE
        litert_cc_api
        absl::flags
        absl::flags_parse
        absl::random_random
        absl::log
        absl::log_internal_check_op
        absl::string_view
        absl::span
)
