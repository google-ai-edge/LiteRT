#!/usr/bin/env bash
# Copyright 2024 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -ex

# Run this script under the root directory.

EXPERIMENTAL_TARGETS_ONLY="${EXPERIMENTAL_TARGETS_ONLY:-false}"
PUBLIC_CACHE_PUSH="${PUBLIC_CACHE_PUSH:-false}"
PUBLIC_CACHE="${PUBLIC_CACHE:-false}"
TEST_LANG_FILTERS="${TEST_LANG_FILTERS:-cc,py}"

BUILD_FLAGS=("-c" "opt"
    "--cxxopt=--std=c++17"
    # add the following flag to avoid clang undefine symbols
    "--copt=-Wno-gnu-offsetof-extensions"
    "--build_tests_only"
    "--test_output=errors"
    "--verbose_failures=true"
    "--test_summary=short"
    "--test_tag_filters=-no_oss,-oss_serial,-gpu,-tpu,-benchmark-test,-v1only"
    "--build_tag_filters=-no_oss,-oss_serial,-gpu,-tpu,-benchmark-test,-v1only"
    "--test_lang_filters=${TEST_LANG_FILTERS}"
    "--flaky_test_attempts=3"
    # Re-enable the following when the compiler supports AVX_VNNI
    "--define=xnn_enable_avxvnni=false"
    "--define=xnn_enable_avx256vnni=false"
    # Re-enable the following when the compiler supports AVX512-AMX
    "--define=xnn_enable_avx512amx=false"
    # Re-enable the foolowing when the compiler supports AVX512_FP16 (clang > 15,
    # GCC > 13)
    "--define=xnn_enable_avx512fp16=false"
    "--nocheck_visibility"
    "--show_timestamps"
    "--experimental_ui_max_stdouterr_bytes=3145728"
  )

# Add Bazel --config flags based on kokoro injected env ie. --config=public_cache
BUILD_FLAGS += ( ${BAZEL_CONFIG_FLAGS} )

# TODO: (b/381310257) - Investigate failing test not included in cpu_full
# TODO: (b/381110338) - Clang errors
# TODO: (b/381124292) - Ambiguous operator errors
# TODO: (b/380870133) - Duplicate op error due to tf_gen_op_wrapper_py
# TODO: (b/382122737) - Module 'keras.src.backend' has no attribute 'convert_to_numpy'
# TODO: (b/382123188) - No member named 'ConvertGenerator' in namespace 'testing'
# TODO: (b/382123664) - Undefined reference due to --no-allow-shlib-undefined: google::protobuf::internal
# TODO(b/385356261): no matching constructor for initialization of 'litert::Tensor::TensorUse'
# TODO(b/385360853): Qualcomm related tests do not build in LiteRT
# TODO(b/385361335): sb_api.h file not found
# TODO(b/385361755): no member named 'view' in 'std::basic_stringstream<char>'
EXCLUDED_TARGETS=(
        "-//tflite/delegates/flex:buffer_map_test"
        "-//tflite/delegates/xnnpack:reduce_test"
        "-//tflite/kernels/variants/py:end_to_end_test"
        "-//tflite/profiling:memory_info_test"
        "-//tflite/profiling:profile_summarizer_test"
        "-//tflite/profiling:profile_summary_formatter_test"
        "-//tflite/python/authoring:authoring_test"
        "-//tflite/python/metrics:metrics_wrapper_test"
        "-//tflite/python:lite_flex_test"
        "-//tflite/python:lite_test"
        "-//tflite/python:lite_v2_test"
        "-//tflite/python:util_test"
        "-//tflite/testing:zip_test_fully_connected_4bit_hybrid_forward-compat_xnnpack"
        "-//tflite/testing:zip_test_fully_connected_4bit_hybrid_mlir-quant_xnnpack"
        "-//tflite/testing:zip_test_fully_connected_4bit_hybrid_with-flex_xnnpack"
        "-//tflite/testing:zip_test_fully_connected_4bit_hybrid_xnnpack"
        "-//tflite/testing:zip_test_depthwiseconv_with-flex"
        "-//tflite/testing:zip_test_depthwiseconv_forward-compat"
        "-//tflite/testing:zip_test_depthwiseconv_mlir-quant"
        "-//tflite/testing:zip_test_depthwiseconv"
        "-//tflite/tools/optimize/debugging/python:debugger_test"
        "-//tflite/tools:convert_image_to_csv_test"
        # Exclude dir which shouldnt run
        "-//tflite/java/..."
        "-//tflite/tools/benchmark/experimental/..."
        "-//tflite/experimental/..."
        "-//tflite/delegates/gpu/..."
)

EXCLUDED_EXPERIMENTAL_TARGETS=(
        "-//tflite/experimental/acceleration/mini_benchmark:fb_storage_test"
        "-//tflite/experimental/litert/c:litert_c_api_common_test"
        "-//tflite/experimental/litert/c:litert_compiled_model_test"
        "-//tflite/experimental/litert/cc:litert_compiled_model_test"
        "-//tflite/experimental/litert/cc:litert_model_predicates_test"
        "-//tflite/experimental/litert/cc:litert_model"
        "-//tflite/experimental/litert/cc:litert_model_test"
        "-//tflite/experimental/litert/cc:litert_tensor_buffer_requirements_test"
        "-//tflite/experimental/litert/cc:litert_tensor_buffer_test"
        "-//tflite/experimental/litert/compiler/plugin:algo"
        "-//tflite/experimental/litert/compiler/plugin:algo_test"
        "-//tflite/experimental/litert/core:byte_code_util_test"
        "-//tflite/experimental/litert/core/model:model_buffer_test"
        "-//tflite/experimental/litert/core/model:model_file_test"
        "-//tflite/experimental/litert/core/util:flatbuffer_tools_test"
        "-//tflite/experimental/litert/runtime:compiled_model_test"
        "-//tflite/experimental/litert/runtime/compiler:jit_compilation_mediatek_test"
        "-//tflite/experimental/litert/runtime/compiler:jit_compilation_qualcomm_test"
        "-//tflite/experimental/litert/runtime/dispatch:dispatch_delegate_google_tensor_test"
        "-//tflite/experimental/litert/runtime/dispatch:dispatch_delegate_mediatek_test"
        "-//tflite/experimental/litert/runtime/dispatch:dispatch_delegate_qualcomm_test"
        "-//tflite/experimental/litert/tools:apply_plugin_test"
        "-//tflite/experimental/litert/tools:dump_test"
        "-//tflite/experimental/litert/tools:tool_display_test"
        "-//tflite/experimental/litert/vendors/cc:convert_graph_test"
        "-//tflite/experimental/litert/vendors/cc:partition_with_capabilities_test"
        "-//tflite/experimental/litert/vendors/examples:example_conversion_impl_test"
        "-//tflite/experimental/litert/vendors/examples:example_plugin_test"
        "-//tflite/experimental/litert/vendors/examples:example_plugin_with_conversions_test"
        "-//tflite/experimental/litert/vendors/google_tensor/dispatch:dispatch_api_google_tensor_test"
        "-//tflite/experimental/litert/vendors/mediatek/dispatch:dispatch_api_mediatek_test"
        "-//tflite/experimental/microfrontend:audio_microfrontend_op_test"
        "-//tflite/experimental/shlo/ops:abs_test"
        "-//tflite/experimental/shlo/ops:binary_elementwise_test"
        "-//tflite/experimental/shlo/ops:cbrt_test"
        "-//tflite/experimental/shlo/ops:ceil_test"
        "-//tflite/experimental/shlo/ops:cosine_test"
        "-//tflite/experimental/shlo/ops:exponential_minus_one_test"
        "-//tflite/experimental/shlo/ops:exponential_test"
        "-//tflite/experimental/shlo/ops:floor_test"
        "-//tflite/experimental/shlo/ops:log_plus_one_test"
        "-//tflite/experimental/shlo/ops:log_test"
        "-//tflite/experimental/shlo/ops:logistic_test"
        "-//tflite/experimental/shlo/ops:negate_test"
        "-//tflite/experimental/shlo/ops:sign_test"
        "-//tflite/experimental/shlo/ops:sine_test"
        "-//tflite/experimental/shlo/ops:sqrt_test"
        "-//tflite/experimental/shlo/ops:tanh_test"
        "-//tflite/experimental/shlo/ops:unary_elementwise_test"
)

if [ "$EXPERIMENTAL_TARGETS_ONLY" == "true" ]; then
    bazel test "${BUILD_FLAGS[@]}" -- //tflite/experimental/... "${EXCLUDED_EXPERIMENTAL_TARGETS[@]}"
else
    bazel test "${BUILD_FLAGS[@]}" -- //tflite/... "${EXCLUDED_TARGETS[@]}"
fi
