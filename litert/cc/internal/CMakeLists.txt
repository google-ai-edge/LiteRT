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

# LiteRT C++ Internal Implementation
cmake_minimum_required(VERSION 3.20)

set(LITERT_CC_INTERNAL_HEADERS
    litert_builder.h
    litert_c_types_printing.h
    litert_context_wrapper.h
    litert_compiled_model_next.h
    litert_consts.h
    litert_detail.h
    litert_dispatch_delegate.h
    litert_extended_model.h
    litert_handle.h
    litert_logging.h
    litert_matchers.h
    litert_model_predicates.h
    litert_numerics.h
    litert_op_options.h
    litert_opaque_options_wrapper.h
    litert_options_wrapper.h
    litert_platform_support.h
    litert_rng.h
    litert_runtime_builtin.h
    litert_runtime_proxy.h
    litert_shared_library.h
    litert_source_location.h
    litert_tensor_buffer_utils.h
    litert_tflite_error_status_builder.h
    scoped_file.h
    scoped_weight_source.h
)

set(LITERT_CC_INTERNAL_MAIN_SRCS
    scoped_file.cc
    litert_op_options.cc
    litert_builder.cc
    litert_shared_library.cc
    litert_extended_model.cc
    litert_runtime_builtin.cc
)

if(WIN32)
    list(APPEND LITERT_CC_INTERNAL_MAIN_SRCS scoped_file_win.cc)
else()
    list(APPEND LITERT_CC_INTERNAL_MAIN_SRCS scoped_file_posix.cc)
endif()

set(LITERT_CC_INTERNAL_EXTRA_SRCS
    litert_compiled_model_next.cc
    litert_model_predicates.cc
    litert_rng.cc
    litert_runtime_no_builtin.cc
)

set(LITERT_CC_INTERNAL_ALL_SRCS
    ${LITERT_CC_INTERNAL_MAIN_SRCS}
    ${LITERT_CC_INTERNAL_EXTRA_SRCS}
)

file(GLOB LITERT_CC_INTERNAL_DISCOVERED_HEADERS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.h"
)
list(SORT LITERT_CC_INTERNAL_DISCOVERED_HEADERS)
set(_litert_cc_internal_expected_headers ${LITERT_CC_INTERNAL_HEADERS})
list(SORT _litert_cc_internal_expected_headers)
if(NOT LITERT_CC_INTERNAL_DISCOVERED_HEADERS STREQUAL _litert_cc_internal_expected_headers)
    message(FATAL_ERROR
        "litert/cc/internal header manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_CC_INTERNAL_DISCOVERED_HEADERS}\n"
        "Expected:\n  ${_litert_cc_internal_expected_headers}\n"
        "Update litert/cc/internal/CMakeLists.txt.")
endif()

file(GLOB LITERT_CC_INTERNAL_DISCOVERED_SRCS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.cc"
)
list(FILTER LITERT_CC_INTERNAL_DISCOVERED_SRCS EXCLUDE REGEX "_test\\.cc$")
list(FILTER LITERT_CC_INTERNAL_DISCOVERED_SRCS EXCLUDE REGEX "^test_shared_library\\.cc$")
if(WIN32)
    list(FILTER LITERT_CC_INTERNAL_DISCOVERED_SRCS EXCLUDE REGEX "^scoped_file_posix\\.cc$")
else()
    list(FILTER LITERT_CC_INTERNAL_DISCOVERED_SRCS EXCLUDE REGEX "^scoped_file_win\\.cc$")
endif()
list(SORT LITERT_CC_INTERNAL_DISCOVERED_SRCS)

if(LITERT_CC_INTERNAL_MAIN_SRCS)
    add_library(litert_cc_internal STATIC
        ${LITERT_CC_INTERNAL_MAIN_SRCS}
    )
else()
    add_library(litert_cc_internal STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../c/empty.cc
    )
endif()

target_include_directories(litert_cc_internal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_cc_internal
    PUBLIC
        litert_c_api
        litert_cc_api
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        absl::synchronization
        flatbuffers::flatbuffers
        Threads::Threads
)

if(LITERT_CC_INTERNAL_EXTRA_SRCS)
    add_library(litert_cc_internal_extra STATIC
        ${LITERT_CC_INTERNAL_EXTRA_SRCS}
    )
else()
    add_library(litert_cc_internal_extra STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../c/empty.cc
    )
endif()

target_include_directories(litert_cc_internal_extra
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
)

target_link_libraries(litert_cc_internal_extra
    PUBLIC
        litert_c_api
        litert_cc_api
        absl::status
        absl::statusor
        absl::strings
        absl::span
        absl::flat_hash_map
        absl::synchronization
        flatbuffers::flatbuffers
        Threads::Threads
)

add_dependencies(litert_cc_internal litert_cc_internal_extra)

# Platform-specific configurations
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(litert_cc_internal PUBLIC ${LOG_LIB})
    target_link_libraries(litert_cc_internal_extra PUBLIC ${LOG_LIB})
endif()
