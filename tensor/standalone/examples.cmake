# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Gemma4 support for the standalone tensor build. SentencePiece
# uses its bundled protobuf-lite and generated sources, including on Android.
# No LiteRT runtime, host protoc, Perfetto, or TFLite weight-cache library is needed.
FetchContent_Declare(litert_sentencepiece
  URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz
  URL_HASH SHA256=9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER internal CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER internal CACHE STRING "" FORCE)
# SentencePiece 0.2.0 declares CMake 3.1 compatibility; allow configuring it
# with CMake 4, which removed policies older than 3.5.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(litert_sentencepiece)
# The 0.2.0 public header uses uint32_t without including cstdint. Newer
# standard libraries no longer provide that declaration through other headers.
target_compile_options(sentencepiece-static PRIVATE
                       "$<$<COMPILE_LANGUAGE:CXX>:-include;cstdint>")
# SentencePiece's bundled flag shim otherwise exports the same two global flag
# names as real Abseil. Keep its internal flags private to this dependency.
target_compile_definitions(sentencepiece-static PRIVATE
  FLAGS_help=SPM_FLAGS_help FLAGS_version=SPM_FLAGS_version)
set_property(DIRECTORY "${litert_sentencepiece_SOURCE_DIR}"
             PROPERTY EXCLUDE_FROM_ALL TRUE)

add_library(litert_gemma_tokenizer STATIC "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma3/tokenizer.cc")
target_include_directories(litert_gemma_tokenizer PUBLIC
                           "${litert_sentencepiece_SOURCE_DIR}/src")
target_link_libraries(litert_gemma_tokenizer PUBLIC
                      litert_tensor_core sentencepiece-static)
if(ANDROID)
  target_link_libraries(litert_gemma_tokenizer PUBLIC log)
endif()

add_library(litert_safetensor_loader STATIC
  "${LITERT_TENSOR_SOURCE_DIR}/examples/utils/minijson.cc" "${LITERT_TENSOR_SOURCE_DIR}/examples/utils/safetensors.cc" "${LITERT_TENSOR_SOURCE_DIR}/examples/utils/safetensor_loader.cc")
target_link_libraries(litert_safetensor_loader PUBLIC litert_tensor_core)

add_library(litert_gemma4_graph STATIC
  "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/gemma4_weights.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/helpers/quantized_embedding.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/helpers/float_activation_fully_connected.cc"
  "${LITERT_TENSOR_SOURCE_DIR}/examples/ops/transformer/transformer_ops_xnnpack.cc")
target_link_libraries(litert_gemma4_graph PUBLIC litert_tensor_xnnpack_runner)

foreach(component config graph)
  litert_tensor_add_test(litert_gemma4_${component}_test
    "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/gemma4_${component}_test.cc" litert_gemma4_graph LABELS gemma4)
endforeach()
target_compile_definitions(litert_gemma4_graph_test PRIVATE LITERT_TENSOR_STANDALONE=1)
foreach(component attention feed_forward_network float_activation_fully_connected mobile_fully_connected quantized_embedding rmsnorm rope transformer)
  litert_tensor_add_test(litert_gemma4_${component}_test
    "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/helpers/${component}_test.cc" litert_gemma4_graph LABELS gemma4)
endforeach()
litert_tensor_add_test(litert_safetensor_loader_test
  "${LITERT_TENSOR_SOURCE_DIR}/examples/utils/safetensor_loader_test.cc" litert_safetensor_loader LABELS gemma4 loader)

if(LITERT_TENSOR_BUILD_NATIVE)
  add_library(litert_gemma4_native STATIC
    "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/native/model/helpers/int8_kv_cache.cc")
  target_link_libraries(litert_gemma4_native PUBLIC
    litert_gemma4_graph litert_safetensor_loader)
  add_executable(gemma4_native_runner
    "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/native/driver.cc")
  target_compile_definitions(gemma4_native_runner PRIVATE
    ABSL_FLAGS_STRIP_NAMES=0 ABSL_FLAGS_STRIP_HELP=0)
  target_link_libraries(gemma4_native_runner PRIVATE
    litert_gemma4_native
    absl::flags absl::flags_parse absl::log_initialize absl::log_globals
    absl::time absl::synchronization)
  set_target_properties(gemma4_native_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  foreach(component stage_runner_smoke stage_workspace active_kv_bank active_attention active_extent_rounding static_int2)
    litert_tensor_add_test(litert_gemma4_native_${component}_test
      "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/native/tests/${component}_test.cc"
      litert_gemma4_native FRAMEWORK exit_code LABELS gemma4 native
      LIBRARIES absl::flags absl::flags_parse absl::log_initialize absl::log_globals absl::time)
  endforeach()
  # This offline weight audit requires a model path; do not put it in CTest.
  add_executable(litert_gemma4_native_static_int2_loader_test
    "${LITERT_TENSOR_SOURCE_DIR}/examples/gemma4/native/tests/static_int2_loader_test.cc")
  target_link_libraries(litert_gemma4_native_static_int2_loader_test PRIVATE
    litert_gemma4_native absl::flags absl::flags_parse absl::log_initialize absl::time)
  set_target_properties(litert_gemma4_native_static_int2_loader_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
