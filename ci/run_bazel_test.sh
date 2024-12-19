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
    # TODO(ecalubaquib): Remove once all the visibility cl's are submitted.
    "--nocheck_visibility"
  )

# TODO(ecalubaquib): Remove the following once the tests are fixed.
# The list of the following bugs are as follows:
# TODO(b/381310257): Investigate failing test not included in cpu_full
# TODO(b/381110338, b/381124292): Clang and ambiguous operator errors
# TODO(b/380870133): Duplicate op error due to tf_gen_op_wrapper_py
# TODO(b/382122737): Module 'keras.src.backend' has no attribute 'convert_to_numpy'
# TODO(b/382123188): No member named 'ConvertGenerator' in namespace 'testing'
# TODO(b/382123664): Undefined reference due to --no-allow-shlib-undefined: google::protobuf::internal
FAILING_TEST=(
        "-//tflite/delegates/flex:buffer_map_test"
        "-//tflite/delegates/gpu/..."
        "-//tflite/delegates/xnnpack:reduce_test"
        "-//tflite/experimental/acceleration/mini_benchmark:fb_storage_test"
        "-//tflite/experimental/litert/c:litert_c_api_common_test"
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
        "-//tflite/tools/optimize/debugging/python:debugger_test"
        "-//tflite/tools:convert_image_to_csv_test"
)

bazel test "${BUILD_FLAGS[@]}" -- //tflite/... "${FAILING_TEST[@]}"
