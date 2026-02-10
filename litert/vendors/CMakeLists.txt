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
include(CheckIncludeFileCXX)
include(FetchContent)

set(NEUROPILOT_HEADERS_DIR "" CACHE PATH "Path to pre-downloaded NeuroPilot headers (root containing neuron/api)")
set(QAIRT_HEADERS_DIR "" CACHE PATH "Path to pre-downloaded QAIRT/QNN headers (root containing Qnn*.h)")

if(NOT NEUROPILOT_HEADERS_DIR)
  if(NOT NEUROPILOT_HEADERS_URL AND NOT NEUROPILOT_GIT_REPO)
    set(NEUROPILOT_HEADERS_URL "https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/66f2c33a-2005-4f0b-afef-2053c8654e4f.gz")
  endif()
  if(NEUROPILOT_HEADERS_URL OR NEUROPILOT_GIT_REPO)
    if(NEUROPILOT_HEADERS_URL)
      FetchContent_Declare(neuro_pilot
        URL ${NEUROPILOT_HEADERS_URL}
      )
    else()
      FetchContent_Declare(neuro_pilot
        GIT_REPOSITORY ${NEUROPILOT_GIT_REPO}
        GIT_TAG        ${NEUROPILOT_GIT_TAG}
      )
    endif()
    FetchContent_MakeAvailable(neuro_pilot)
    set(NEUROPILOT_HEADERS_DIR ${neuro_pilot_SOURCE_DIR})
    set(_np_candidates)
    list(APPEND _np_candidates
      "${NEUROPILOT_HEADERS_DIR}/${NEUROPILOT_HEADERS_SUBDIR}"
      "${NEUROPILOT_HEADERS_DIR}/v8_latest/host/include"
      "${NEUROPILOT_HEADERS_DIR}/v9_latest/host/include"
      "${NEUROPILOT_HEADERS_DIR}/v7_latest/host/include"
      "${NEUROPILOT_HEADERS_DIR}/v8_0_8/host/include"
      "${NEUROPILOT_HEADERS_DIR}/host/include"
      "${NEUROPILOT_HEADERS_DIR}/include"
    )
    foreach(_cand IN LISTS _np_candidates)
      if(_cand AND EXISTS "${_cand}/neuron/api/NeuronAdapter.h")
        set(NEUROPILOT_HEADERS_DIR "${_cand}")
        break()
      endif()
    endforeach()
    message(STATUS "NeuroPilot headers at: ${NEUROPILOT_HEADERS_DIR}")
  endif()
endif()

# LINT.IfChange(qairt_headers_dir)
if(NOT QAIRT_HEADERS_DIR)
  if(NOT QAIRT_HEADERS_URL AND NOT QAIRT_GIT_REPO)
    set(QAIRT_HEADERS_URL "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip")
    set(QAIRT_HEADERS_SUBDIR "qairt/2.42.0.251225")
  endif()
  if(QAIRT_HEADERS_URL)
    set(_qnn_archive "${CMAKE_BINARY_DIR}/_deps/qnn_headers.zip")
    file(DOWNLOAD "${QAIRT_HEADERS_URL}" "${_qnn_archive}" SHOW_PROGRESS)
    set(_qnn_extract_dir "${CMAKE_BINARY_DIR}/_deps/qnn_headers")
    file(MAKE_DIRECTORY "${_qnn_extract_dir}")
    file(ARCHIVE_EXTRACT INPUT "${_qnn_archive}" DESTINATION "${_qnn_extract_dir}")
    set(QAIRT_HEADERS_DIR "${_qnn_extract_dir}")
  elseif(QAIRT_GIT_REPO)
    FetchContent_Declare(qairt_headers GIT_REPOSITORY ${QAIRT_GIT_REPO} GIT_TAG ${QAIRT_GIT_TAG})
    FetchContent_MakeAvailable(qairt_headers)
    set(QAIRT_HEADERS_DIR ${qairt_headers_SOURCE_DIR})
  endif()
  if(QAIRT_HEADERS_SUBDIR)
    set(QAIRT_HEADERS_DIR ${QAIRT_HEADERS_DIR}/${QAIRT_HEADERS_SUBDIR})
  endif()
  set(_qnn_candidates)
  list(APPEND _qnn_candidates
    "${QAIRT_HEADERS_DIR}/include"
    "${QAIRT_HEADERS_DIR}"
  )
  foreach(_cand IN LISTS _qnn_candidates)
    if(_cand AND EXISTS "${_cand}/QnnCommon.h")
      set(QAIRT_HEADERS_DIR "${_cand}")
      break()
    elseif(_cand AND EXISTS "${_cand}/QNN/QnnCommon.h")
      set(QAIRT_HEADERS_DIR "${_cand}/QNN")
      break()
    endif()
  endforeach()
  message(STATUS "QAIRT/QNN headers at: ${QAIRT_HEADERS_DIR}")
endif()

# LINT.ThenChange(
#     ../../opensource_only/ci/tools/python/vendor_sdk/qualcomm/setup.py:wheel_qairt_sdk_version,
#     ../google/npu_runtime_libraries/fetch_qualcomm_library.sh:fetch_qairt_sdk_version,
#     ../google/npu_runtime_libraries/fetch_qualcomm_library_jit.sh:fetch_qairt_sdk_version,
# )

if(NEUROPILOT_HEADERS_DIR)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${NEUROPILOT_HEADERS_DIR})
endif()
if(QAIRT_HEADERS_DIR)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${QAIRT_HEADERS_DIR})
endif()

function(_litert_add_dispatch_so VENDOR DIR OUTPUT_BASENAME)
  set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}")
  if(NOT EXISTS "${SRC_DIR}/dispatch_api.cc")
    message(STATUS "Skipping ${VENDOR} dispatch: ${SRC_DIR}/dispatch_api.cc not found")
    return()
  endif()

  file(GLOB DISPATCH_SRCS
    "${SRC_DIR}/dispatch_api.cc"
    "${SRC_DIR}/*dispatch_device_context.cc"
    "${SRC_DIR}/*dispatch_graph.cc"
    "${SRC_DIR}/*dispatch_invocation_context.cc"
    "${SRC_DIR}/sb_api_late_binding.cc"
    "${SRC_DIR}/device_context.cc"
    "${SRC_DIR}/invocation_context.cc"
  )

  if(VENDOR STREQUAL "GoogleTensor")
    list(APPEND DISPATCH_SRCS
      "${SRC_DIR}/dispatch_api_config.cc"
      "${SRC_DIR}/dispatch_api_utils.cc"
    )
  endif()

  if(VENDOR STREQUAL "MediaTek")
    set(FLATC_EXECUTABLE "")

    if(TFLITE_HOST_TOOLS_DIR AND EXISTS "${TFLITE_HOST_TOOLS_DIR}/flatc")
      set(FLATC_EXECUTABLE "${TFLITE_HOST_TOOLS_DIR}/flatc")
    elseif(TFLITE_BUILD_DIR AND EXISTS "${TFLITE_BUILD_DIR}/host_flatc/_deps/flatbuffers-build/flatc")
      set(FLATC_EXECUTABLE "${TFLITE_BUILD_DIR}/host_flatc/_deps/flatbuffers-build/flatc")
    endif()

    if(NOT FLATC_EXECUTABLE)
      find_program(FLATC_EXECUTABLE NAMES flatc)
    endif()
    if(NOT FLATC_EXECUTABLE)
      message(STATUS "flatc not found in PATH; fetching FlatBuffers to build flatc")
      FetchContent_Declare(flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG v25.9.23
      )
      set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
      FetchContent_MakeAvailable(flatbuffers)
      if(NOT TARGET flatc)
        message(FATAL_ERROR "FlatBuffers 'flatc' target not available after fetch")
      endif()
      set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
    endif()
    set(_mtk_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include/litert/vendors/mediatek/schema")
    file(MAKE_DIRECTORY "${_mtk_gen_dir}")
    set(_mtk_schema "${CMAKE_CURRENT_SOURCE_DIR}/mediatek/schema/neuron_schema.fbs")
    set(_mtk_gen_hdr "${_mtk_gen_dir}/neuron_schema_generated.h")
    add_custom_command(
      OUTPUT "${_mtk_gen_hdr}"
      COMMAND ${FLATC_EXECUTABLE} --cpp -o "${_mtk_gen_dir}" "${_mtk_schema}"
      DEPENDS "${_mtk_schema}"
      COMMENT "Generating MediaTek neuron_schema_generated.h with flatc"
      VERBATIM)
    add_custom_target(mediatek_schema_gen DEPENDS "${_mtk_gen_hdr}")
    list(APPEND DISPATCH_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mediatek/neuron_adapter_api.cc")
  endif()

  if(VENDOR STREQUAL "Qualcomm")
    list(APPEND DISPATCH_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/qnn_manager.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/context_binary_info.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/common.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/utils/miscs.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/schema/soc_table.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/backends/qnn_backend.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/backends/htp_backend.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/backends/htp_perf_control.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/backends/ir_backend.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/wrappers/quantize_params_wrapper.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/wrappers/tensor_wrapper.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/wrappers/param_wrapper.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/wrappers/op_wrapper.cc"
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND DISPATCH_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/utils/log_android.cc")
    else()
      list(APPEND DISPATCH_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/utils/log_default.cc")
    endif()
  endif()

  set(TGT "dispatch_api_${VENDOR}_so")
  add_library(${TGT} SHARED ${DISPATCH_SRCS})
  if(VENDOR STREQUAL "MediaTek")
    add_dependencies(${TGT} mediatek_schema_gen)
    target_include_directories(${TGT} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
  endif()
  target_include_directories(${TGT}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>
      $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
      $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${DIR}>
      $<$<BOOL:${NEUROPILOT_HEADERS_DIR}>:${NEUROPILOT_HEADERS_DIR}>
      $<$<BOOL:${QAIRT_HEADERS_DIR}>:${QAIRT_HEADERS_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qualcomm>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/utils>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/wrappers>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/backends>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qualcomm/core/schema>
  )
  target_link_libraries(${TGT}
    PUBLIC
      litert_runtime_c_api_static
      absl::algorithm
      absl::base
      absl::cleanup
      absl::flat_hash_map
      absl::flat_hash_set
      absl::str_format
      absl::strings
      absl::string_view
      absl::span
      absl::status
      absl::statusor)
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${TGT} PRIVATE android log)
  endif()
  set_target_properties(${TGT} PROPERTIES
    OUTPUT_NAME "LiteRtDispatch_${OUTPUT_BASENAME}"
    POSITION_INDEPENDENT_CODE ON
  )

  if(UNIX AND NOT APPLE)
    target_link_options(${TGT} PRIVATE "-Wl,-soname=libLiteRtDispatch_${OUTPUT_BASENAME}.so")
  endif()
endfunction()

# Google Tensor
set(GOOGLE_TENSOR_SB_API_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/google_tensor/dispatch/sb_api.h")
if(EXISTS "${GOOGLE_TENSOR_SB_API_HEADER}")
  _litert_add_dispatch_so(GoogleTensor "google_tensor/dispatch" "GoogleTensor")
else()
  message(STATUS "Skipping Google Tensor dispatch: sb_api.h not found")
endif()
# MediaTek
check_include_file_cxx("neuron/api/NeuronAdapter.h" HAVE_MEDIATEK_NEURON_HDR)
if(NEUROPILOT_HEADERS_DIR AND EXISTS "${NEUROPILOT_HEADERS_DIR}/neuron/api/NeuronAdapter.h")
  set(HAVE_MEDIATEK_NEURON_HDR TRUE)
endif()
if(HAVE_MEDIATEK_NEURON_HDR)
  _litert_add_dispatch_so(MediaTek "mediatek/dispatch" "MediaTek")
else()
  message(STATUS "Skipping MediaTek dispatch: neuron/api/NeuronAdapter.h not found")
endif()
# Intel OpenVINO
check_include_file_cxx("openvino/openvino.hpp" HAVE_OPENVINO_HDR)
if(HAVE_OPENVINO_HDR)
  _litert_add_dispatch_so(OpenVINO "intel_openvino/dispatch" "OpenVINO")
else()
  message(STATUS "Skipping OpenVINO dispatch: openvino/openvino.hpp not found")
endif()

# Qualcomm
option(LITERT_ENABLE_QUALCOMM "Enable Qualcomm dispatch build (requires QNN SDK headers/libs)" OFF)
if(QAIRT_HEADERS_DIR AND NOT LITERT_ENABLE_QUALCOMM)
  message(STATUS "Enabling Qualcomm dispatch: QNN headers detected at ${QAIRT_HEADERS_DIR}")
  set(LITERT_ENABLE_QUALCOMM ON CACHE BOOL "" FORCE)
endif()
if(LITERT_ENABLE_QUALCOMM)
  # Use modular CMake structure for Qualcomm
  add_subdirectory(qualcomm)
endif()
