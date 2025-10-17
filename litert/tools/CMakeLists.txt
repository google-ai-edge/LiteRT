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

# Tool display
add_library(litert_tool_display STATIC
    tool_display.cc
)

target_include_directories(litert_tool_display
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tool_display
    PUBLIC
        litert_cc_api
        litert_cc_internal
        absl::strings
        absl::string_view
        absl::span
        absl::status
        absl::statusor
)

# LiteRT Dump
add_library(litert_dump STATIC
    dump.cc
)

target_include_directories(litert_dump
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_dump
    PUBLIC
        litert_core
        litert_core_model
        tensorflow-lite
        absl::strings
        absl::status
        absl::statusor
)

# Apply plugin
add_library(litert_apply_plugin STATIC
    apply_plugin.cc
)

target_include_directories(litert_apply_plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_apply_plugin
    PUBLIC
        litert_core
        litert_core_model
        absl::strings
        absl::status
        absl::statusor
)

# NPU numerics check
add_library(litert_npu_numerics_check STATIC
    npu_numerics_check.cc
)

target_include_directories(litert_npu_numerics_check
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_npu_numerics_check
    PUBLIC
        litert_cc_api
        litert_runtime
        absl::strings
        absl::status
        absl::statusor
)

# Add flags
add_subdirectory(flags)

# run_model
add_executable(run_model
    run_model.cc
)

target_include_directories(run_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

target_link_libraries(run_model
    PRIVATE
        litert_tool_display
        litert_c_api
        litert_cc_api
        litert_cc_internal
        litert_cc_options
        litert_core
        litert_core_model
        litert_runtime
        tensorflow-lite
        litert_tool_flags_google_tensor
        litert_tool_flags_intel_openvino
        litert_tool_flags_mediatek
        litert_tool_flags_qualcomm
        absl::flags
        absl::flags_parse
        absl::random_random
        absl::log
        absl::log_internal_check_op
        absl::string_view
        absl::span
        Threads::Threads
)

# analyze_model
add_executable(analyze_model
    analyze_model_main.cc
)

target_include_directories(analyze_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

target_link_libraries(analyze_model
    PRIVATE
        litert_dump
        litert_tool_display
        litert_core
        litert_core_model
        tensorflow-lite
        absl::flags
        absl::flags_parse
        absl::log
        absl::status
        absl::statusor
)

# apply_plugin
add_executable(apply_plugin_main
    apply_plugin_main.cc
)

target_include_directories(apply_plugin_main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

target_link_libraries(apply_plugin_main
    PRIVATE
        litert_apply_plugin
        litert_dump
        litert_tool_flags_apply_plugin
        litert_tool_flags_common
        litert_tool_flags_google_tensor
        litert_tool_flags_intel_openvino
        litert_tool_flags_mediatek
        litert_tool_flags_qualcomm

        litert_tool_display
        litert_core
        litert_core_model
        litert_runtime
        tensorflow-lite
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_internal_check_op
        absl::status
        absl::statusor
)

# Platform-specific configurations for all executables
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(run_model PRIVATE ${LOG_LIB})
    target_link_libraries(analyze_model PRIVATE ${LOG_LIB})
    target_link_libraries(apply_plugin_main PRIVATE ${LOG_LIB})
    # Statically link TFLite GPU/GL core objects for tool executables on Android.
    if(TARGET litert_tflite_gpu_gl_core)
        target_link_libraries(run_model PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(apply_plugin_main PRIVATE litert_tflite_gpu_gl_core)
    endif()
endif()

# Install targets
install(TARGETS run_model analyze_model apply_plugin_main
    RUNTIME DESTINATION bin
)
