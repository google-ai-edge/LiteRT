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

# LiteRT tensor_utils
add_library(litert_tensor_utils STATIC
    tensor_utils.cc
)

target_include_directories(litert_tensor_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_tensor_utils
    PRIVATE
        absl::log
        absl::log_internal_message
        absl::span
        absl::string_view
        litert_cc_api
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

# Add flags
add_subdirectory(flags)

# Common libraries for tools that use TFLite
set(LITERT_TOOL_COMMON_LIBS
    litert_c_api
    litert_cc_api
    litert_cc_internal
    litert_cc_options
    litert_compiler_plugin
    litert_core
    litert_core_model
    litert_runtime
    litert_options_parser_registry
)

set(ABSEIL_TOOL_LIBS
    absl::check
    absl::die_if_null
    absl::flags
    absl::flags_parse
    absl::log
    absl::log_internal_check_op
    absl::log_internal_message
    absl::random_random
    absl::string_view
    absl::span
    absl::status
    absl::statusor
)

set(LITERT_TOOL_VENDOR_FLAG_LIBS
    litert_tool_flags_google_tensor
    litert_tool_flags_intel_openvino
    litert_tool_flags_mediatek
    litert_tool_flags_qualcomm
    litert_tool_flags_samsung
)

# NPU numerics check
add_executable(litert_npu_numerics_check
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

if(ANDROID)
    target_link_libraries(litert_npu_numerics_check
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
elseif(APPLE)
    target_link_libraries(litert_npu_numerics_check
        PRIVATE
            litert_tool_display
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
else()
    target_link_libraries(litert_npu_numerics_check
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
            litert_tensor_utils
    )
endif()

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

# absl strips flag names/registration on Android by default (ABSL_FLAGS_STRIP_NAMES=1).
# This tool relies on absl::ParseCommandLine, so we must override that behaviour.
if(ANDROID)
    target_compile_definitions(run_model PRIVATE ABSL_FLAGS_STRIP_NAMES=0)
endif()

# Platform-aware linking
if(ANDROID)
    target_link_libraries(run_model
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
elseif(APPLE)
    target_link_libraries(run_model
        PRIVATE
            litert_tool_display
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
else()
    target_link_libraries(run_model
        PRIVATE
            litert_tool_display
            "-Wl,--whole-archive"
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
            litert_tensor_utils
    )
endif()

# benchmark_tool
find_package(Protobuf REQUIRED)

set(_LITERT_XLA_SOURCE_DIR "${XLA_SOURCE_DIR}")
if(NOT _LITERT_XLA_SOURCE_DIR)
    set(_LITERT_XLA_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/third_party/xla")
endif()

get_filename_component(_LITERT_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH)
set(_LITERT_TOOLS_PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_protos")

set(LITERT_BENCHMARK_PROTO_FILES
    ${_LITERT_SOURCE_ROOT}/tflite/tools/benchmark/proto/benchmark_result.proto
    ${_LITERT_SOURCE_ROOT}/tflite/profiling/proto/profiling_info.proto
    ${_LITERT_SOURCE_ROOT}/tflite/profiling/proto/model_runtime_info.proto
)
set(LITERT_BENCHMARK_PROTO_SRCS
    ${_LITERT_TOOLS_PROTO_GEN_DIR}/tflite/tools/benchmark/proto/benchmark_result.pb.cc
    ${_LITERT_TOOLS_PROTO_GEN_DIR}/tflite/tools/benchmark/proto/benchmark_result.pb.h
    ${_LITERT_TOOLS_PROTO_GEN_DIR}/tflite/profiling/proto/profiling_info.pb.cc
    ${_LITERT_TOOLS_PROTO_GEN_DIR}/tflite/profiling/proto/profiling_info.pb.h
    ${_LITERT_TOOLS_PROTO_GEN_DIR}/tflite/profiling/proto/model_runtime_info.pb.cc
    ${_LITERT_TOOLS_PROTO_GEN_DIR}/tflite/profiling/proto/model_runtime_info.pb.h
)

add_custom_command(
    OUTPUT ${LITERT_BENCHMARK_PROTO_SRCS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS
        --cpp_out=${_LITERT_TOOLS_PROTO_GEN_DIR}
        --proto_path=${_LITERT_SOURCE_ROOT}
        ${LITERT_BENCHMARK_PROTO_FILES}
    DEPENDS
        ${Protobuf_PROTOC_EXECUTABLE}
        ${LITERT_BENCHMARK_PROTO_FILES}
    VERBATIM
)

add_library(litert_benchmark_protos STATIC ${LITERT_BENCHMARK_PROTO_SRCS})
set_source_files_properties(${LITERT_BENCHMARK_PROTO_SRCS}
    PROPERTIES GENERATED TRUE
)
target_include_directories(litert_benchmark_protos
    PUBLIC
        ${_LITERT_TOOLS_PROTO_GEN_DIR}
)
target_link_libraries(litert_benchmark_protos
    PUBLIC
        protobuf::libprotobuf
)

set(LITERT_BENCHMARK_TOOL_SRCS
    benchmark_litert_model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/util/perfetto_profiling.cc
    ${TFLITE_SOURCE_DIR}/profiling/memory_usage_monitor.cc
    ${TFLITE_SOURCE_DIR}/profiling/model_runtime_info.cc
    ${TFLITE_SOURCE_DIR}/tools/benchmark/benchmark_model.cc
    ${TFLITE_SOURCE_DIR}/tools/benchmark/benchmark_utils.cc
    ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
    ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
    ${TFLITE_SOURCE_DIR}/tools/utils.cc
    ${_LITERT_XLA_SOURCE_DIR}/xla/tsl/util/stats_calculator.cc
)

add_library(litert_benchmark_tool_lib STATIC
    ${LITERT_BENCHMARK_TOOL_SRCS}
)

target_include_directories(litert_benchmark_tool_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${_LITERT_XLA_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(litert_benchmark_tool_lib
    PUBLIC
        litert_c_api
        litert_cc_api
        litert_cc_internal
        litert_cc_options
        litert_core
        litert_core_model
        litert_runtime
        litert_options_parser_registry
        litert_benchmark_protos
        protobuf::libprotobuf
        tensorflow-lite
        absl::flat_hash_map
        absl::log
        absl::span
        absl::str_format
        absl::strings
)

add_executable(benchmark_tool
    benchmark_litert_model_main.cc
)

set_target_properties(benchmark_tool PROPERTIES
    OUTPUT_NAME benchmark_model
)

if(ANDROID)
    target_compile_definitions(benchmark_tool PRIVATE ABSL_FLAGS_STRIP_NAMES=0)
endif()

set(LITERT_BENCHMARK_TOOL_COMMON_LIBS
    litert_benchmark_tool_lib
    ${LITERT_TOOL_COMMON_LIBS}
    ${ABSEIL_TOOL_LIBS}
    absl::flags_reflection
    litert_benchmark_protos
    protobuf::libprotobuf
)

if(ANDROID)
    target_link_libraries(benchmark_tool
        PRIVATE
            "-Wl,--whole-archive"
            ${LITERT_TOOL_VENDOR_FLAG_LIBS}
            "-Wl,--no-whole-archive"
            ${LITERT_BENCHMARK_TOOL_COMMON_LIBS}
            tensorflow-lite
            Threads::Threads
    )
elseif(APPLE)
    target_link_libraries(benchmark_tool
        PRIVATE
            ${LITERT_TOOL_VENDOR_FLAG_LIBS}
            ${LITERT_BENCHMARK_TOOL_COMMON_LIBS}
            tensorflow-lite
            Threads::Threads
    )
else()
    target_link_libraries(benchmark_tool
        PRIVATE
            "-Wl,--whole-archive"
            ${LITERT_TOOL_VENDOR_FLAG_LIBS}
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${LITERT_BENCHMARK_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

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

# Platform-aware linking for analyze_model
if(ANDROID OR APPLE)
    target_link_libraries(analyze_model
        PRIVATE
            litert_dump
            litert_tool_display
            litert_c_api
            litert_core
            litert_core_model
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
    )
else()
    target_link_libraries(analyze_model
        PRIVATE
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            litert_dump
            litert_tool_display
            litert_c_api
            litert_core
            litert_core_model
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

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

# Platform-aware linking
if(ANDROID)
    target_link_libraries(apply_plugin_main
        PRIVATE
            litert_tool_display
            litert_apply_plugin
            litert_dump
            "-Wl,--whole-archive"
            litert_tool_flags_apply_plugin
            litert_tool_flags_common
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
        )
elseif(APPLE)
    target_link_libraries(apply_plugin_main
        PRIVATE
            litert_tool_display
            litert_apply_plugin
            litert_dump
            litert_tool_flags_apply_plugin
            litert_tool_flags_common
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
        )
else()
    target_link_libraries(apply_plugin_main
        PRIVATE
            "-Wl,--whole-archive"
            litert_tool_flags_apply_plugin
            litert_tool_flags_common
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            litert_tool_flags_qualcomm
            litert_tool_flags_samsung
            "-Wl,--no-whole-archive"
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            litert_tool_display
            litert_apply_plugin
            litert_dump
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

# extract_bytecode
add_executable(extract_bytecode
    extract_bytecode.cc
)

target_include_directories(extract_bytecode
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

if(ANDROID OR APPLE)
    target_link_libraries(extract_bytecode
        PRIVATE
            litert_tool_display
            litert_cc_internal
            litert_core
            litert_core_model
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
    )
else()
    target_link_libraries(extract_bytecode
        PRIVATE
            Threads::Threads
            "-Wl,--start-group"
            litert_tool_display
            litert_cc_internal
            litert_core
            litert_core_model
            ${ABSEIL_TOOL_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
    )
endif()

# npu_numerics_check
add_executable(npu_numerics_check
    npu_numerics_check.cc
)

target_include_directories(npu_numerics_check
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${TENSORFLOW_SOURCE_DIR}
)

# Platform-aware linking
if(ANDROID OR APPLE)
    target_link_libraries(npu_numerics_check
        PRIVATE
            litert_tool_flags_qualcomm
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            ${LITERT_TOOL_COMMON_LIBS}
            tensorflow-lite
            ${ABSEIL_TOOL_LIBS}
            Threads::Threads
            litert_tensor_utils
    )
else()
    target_link_libraries(npu_numerics_check
        PRIVATE
            litert_tool_flags_qualcomm
            litert_tool_flags_google_tensor
            litert_tool_flags_intel_openvino
            litert_tool_flags_mediatek
            Threads::Threads
            tensorflow-lite
            "-Wl,--start-group"
            ${ABSEIL_TOOL_LIBS}
            ${LITERT_TOOL_COMMON_LIBS}
            "-Wl,--end-group"
            ${CMAKE_DL_LIBS}
            litert_tensor_utils
    )
endif()

# Platform-specific configurations for all executables
if(LITERT_PLATFORM_ANDROID)
    target_link_libraries(run_model PRIVATE ${LOG_LIB})
    target_link_libraries(benchmark_tool PRIVATE ${LOG_LIB})
    target_link_libraries(analyze_model PRIVATE ${LOG_LIB})
    target_link_libraries(apply_plugin_main PRIVATE ${LOG_LIB})
    target_link_libraries(extract_bytecode PRIVATE ${LOG_LIB})
    # Statically link TFLite GPU/GL core objects for tool executables on Android.
    if(TARGET litert_tflite_gpu_gl_core)
        target_link_libraries(npu_numerics_check PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(litert_npu_numerics_check PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(run_model PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(benchmark_tool PRIVATE litert_tflite_gpu_gl_core)
        target_link_libraries(apply_plugin_main PRIVATE litert_tflite_gpu_gl_core)
    endif()
endif()

# Install targets
install(TARGETS run_model benchmark_tool analyze_model apply_plugin_main extract_bytecode
    RUNTIME DESTINATION bin
)
