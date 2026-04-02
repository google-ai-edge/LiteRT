# Copyright (C) 2026 Samsung Electronics Co. LTD.
# SPDX-License-Identifier: Apache-2.0
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

# Samsung Dispatch API Shared Library
cmake_minimum_required(VERSION 3.20)

add_library(dispatch_api_samsung_so SHARED)

target_sources(dispatch_api_samsung_so
    PRIVATE
        dispatch_api.cc
        enn_manager.cc
        litert_dispatch_device_context.cc
        litert_dispatch_invocation_context.cc
)

target_include_directories(dispatch_api_samsung_so
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<$<BOOL:${LITECORE_HEADERS_DIR}>:${LITECORE_HEADERS_DIR}>
)

target_link_libraries(dispatch_api_samsung_so
    PRIVATE
        samsung_soc_model
        samsung_ai_litecore_manager
        litert_runtime_c_api_static
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(dispatch_api_samsung_so PRIVATE android log)
endif()

set_target_properties(dispatch_api_samsung_so PROPERTIES
    OUTPUT_NAME "LiteRtDispatch_Samsung"
)

if(UNIX AND NOT APPLE)
    # Set soname
    target_link_options(dispatch_api_samsung_so PRIVATE
        "-Wl,-soname=libLiteRtDispatch_Samsung.so"
    )

    target_link_libraries(dispatch_api_samsung_so PRIVATE
        "-Wl,--start-group"
        litert_runtime_c_api_shared_lib
        litert_compiler_plugin
        litert_cc_api
        litert_cc_options
        tensorflow-lite
        Threads::Threads
        "-Wl,--end-group"
    )
endif()
