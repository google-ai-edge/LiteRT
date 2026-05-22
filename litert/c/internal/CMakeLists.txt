# Copyright 2026 Google LLC.
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

set(LITERT_C_INTERNAL_HEADERS
    internal/litert_accelerator.h
    internal/litert_accelerator_def.h
    internal/litert_accelerator_registration.h
    internal/litert_compiler_context.h
    internal/litert_custom_tensor_buffer_handlers_def.h
    internal/litert_delegate_wrapper.h
    internal/litert_dispatch_delegate.h
    internal/litert_external_litert_buffer_context.h
    internal/litert_logging.h
    internal/litert_logging_helper.h
    internal/litert_logging_helper_with_compiler_context.h
    internal/litert_logging_helper_with_runtime_context.h
    internal/litert_options_helper.h
    internal/litert_runtime_c_api.h
    internal/litert_runtime_context.h
    internal/litert_scheduling_info.h
    internal/litert_tensor_buffer_registry.h
)

set(LITERT_C_INTERNAL_SOURCES
    internal/litert_accelerator.cc
    internal/litert_accelerator_registration.cc
    internal/litert_compiler_context.cc
    internal/litert_delegate_wrapper.cc
    internal/litert_external_litert_buffer_context.cc
    internal/litert_runtime_context.cc
    internal/litert_tensor_buffer_registry.cc
)

set(LITERT_C_INTERNAL_ALL_SOURCES
    ${LITERT_C_INTERNAL_SOURCES}
    internal/litert_logging.cc
    internal/litert_logging_helper.cc
    internal/litert_logging_helper_with_compiler_context.cc
    internal/litert_logging_helper_with_runtime_context.cc
)

file(GLOB LITERT_C_INTERNAL_DISCOVERED_HEADERS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.h"
)
list(TRANSFORM LITERT_C_INTERNAL_DISCOVERED_HEADERS PREPEND "internal/")
list(SORT LITERT_C_INTERNAL_DISCOVERED_HEADERS)
set(_litert_c_internal_expected_headers ${LITERT_C_INTERNAL_HEADERS})
list(SORT _litert_c_internal_expected_headers)
if(NOT LITERT_C_INTERNAL_DISCOVERED_HEADERS STREQUAL _litert_c_internal_expected_headers)
    message(FATAL_ERROR
        "litert/c/internal header manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_C_INTERNAL_DISCOVERED_HEADERS}\n"
        "Expected:\n  ${_litert_c_internal_expected_headers}\n"
        "Update litert/c/internal/CMakeLists.txt.")
endif()

file(GLOB LITERT_C_INTERNAL_DISCOVERED_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "*.cc"
)
list(TRANSFORM LITERT_C_INTERNAL_DISCOVERED_SOURCES PREPEND "internal/")
list(FILTER LITERT_C_INTERNAL_DISCOVERED_SOURCES EXCLUDE REGEX "_test\\.cc$")
list(SORT LITERT_C_INTERNAL_DISCOVERED_SOURCES)
set(_litert_c_internal_expected_sources ${LITERT_C_INTERNAL_ALL_SOURCES})
list(SORT _litert_c_internal_expected_sources)
if(NOT LITERT_C_INTERNAL_DISCOVERED_SOURCES STREQUAL _litert_c_internal_expected_sources)
    message(FATAL_ERROR
        "litert/c/internal source manifest is out of sync.\n"
        "Discovered:\n  ${LITERT_C_INTERNAL_DISCOVERED_SOURCES}\n"
        "Expected:\n  ${_litert_c_internal_expected_sources}\n"
        "Update litert/c/internal/CMakeLists.txt.")
endif()

set(LITERT_C_INTERNAL_HEADERS ${LITERT_C_INTERNAL_HEADERS} PARENT_SCOPE)
set(LITERT_C_INTERNAL_SOURCES ${LITERT_C_INTERNAL_SOURCES} PARENT_SCOPE)
