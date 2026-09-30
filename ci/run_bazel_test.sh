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
# TODO: b/398924022  remove these once litert is migrated to tflite/
EXPERIMENTAL_TARGETS_ONLY="${EXPERIMENTAL_TARGETS_ONLY:-false}"
LITERT_TARGETS_ONLY="${LITERT_TARGETS_ONLY:-false}"
TEST_LANG_FILTERS="${TEST_LANG_FILTERS:-cc,py}"
# Set to 1 to build with the real TensorFlow and also run the targets that need
# it, e.g. the converter and the tflite/testing tests.
LITERT_WITH_TENSORFLOW="${LITERT_WITH_TENSORFLOW:-0}"

# Common flags for both building and testing.
COMMON_BUILD_FLAGS=(
    "--config=bulk_build_cpu"
    "--keep_going"
    "--repo_env=USE_PYWRAP_RULES=True"
)

# Add Bazel --config flags based on kokoro injected env ie. --config=public_cache
COMMON_BUILD_FLAGS+=(${BAZEL_CONFIG_FLAGS})

# Conditionally use local submodules vs http_archive tf
if [[ "${USE_LOCAL_TF}" == "true" ]]; then
  COMMON_BUILD_FLAGS+=("--config=use_local_tf")
fi

# Flags specific to testing.
TEST_FLAGS=(
    "--config=bulk_test_cpu"
    "--test_lang_filters=${TEST_LANG_FILTERS}"
)

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
EXCLUDED_TARGETS=(
        "-//tflite/delegates/flex:buffer_map_test"
        "-//tflite/delegates/gpu/cl/kernels:convolution_transposed_3x3_test"
        "-//tflite/delegates/xnnpack:reduce_test"
        "-//tflite/experimental/microfrontend:audio_microfrontend_op_test"
        "-//tflite/profiling:memory_info_test"
        "-//tflite/profiling:profile_summarizer_test"
        "-//tflite/profiling:profile_summary_formatter_test"
        "-//tflite/python/metrics:metrics_wrapper_test"
        "-//tflite/python:convert_saved_model_test"
        "-//tflite/python:convert_test"
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
        "-//tflite/testing:zip_test_depthwiseconv"
        "-//tflite/testing:zip_test_depthwiseconv_forward-compat"
        "-//tflite/testing:zip_test_depthwiseconv_mlir-quant"
        "-//tflite/testing:zip_test_depthwiseconv_with-flex"
        # Exclude dir which shouldnt run
        "-//tflite/java/..."
        "-//tflite/delegates/gpu/..."
        "-//tflite/delegates/nnapi/..."
        # Flex delegate is not supported in the OSS build.
        "-//tflite/delegates/flex/..."
        "-//tflite:model_flex_test"
        "-//tflite/tools:list_flex_ops"
        "-//tflite/tools:list_flex_ops_main"
        "-//tflite/tools:list_flex_ops_main_lib"
        "-//tflite/tools:list_flex_ops_test"
        "-//tflite/tools/benchmark:benchmark_model_plus_flex"
        # TODO: (b/410925271) - Targets not migrated to pywrap_rules yet
)

# //tflite targets that need the real TensorFlow (converter, tflite/testing,
# ...). They are not dropped from CI:
# - With LITERT_WITH_TENSORFLOW=1 (Internal CI 'cpu_full', Linux), they are
#   built with --config=with_tensorflow and tested as part of //tflite/...
# - Without it, they are added to EXCLUDED_TARGETS because the default build
#   has no TensorFlow.
TF_EXCLUDED_TARGETS=(
        "-//tflite/converter/..."
        "-//tflite/kernels/parse_example/..."
        "-//tflite/testing/..."
        "-//tflite/tools/optimize/debugging/python/..."
        "-//tflite:simple_planner_test"
        "-//tflite/core/tools:verifier_test"
        "-//tflite/experimental/microfrontend:audio_microfrontend_op_lib"
        "-//tflite/profiling:subgraph_tensor_profiler"
        "-//tflite/profiling:subgraph_tensor_profiler_test"
        "-//tflite/python:analyzer"
        "-//tflite/python:analyzer_test"
        "-//tflite/python:convert"
        "-//tflite/python:convert_file_to_c_source"
        "-//tflite/python:convert_file_to_c_source_test"
        "-//tflite/python:convert_saved_model"
        "-//tflite/python:interpreter_test"
        "-//tflite/python:lite"
        "-//tflite/python:lite_constants"
        "-//tflite/python:lite_v2_test_util"
        "-//tflite/python:op_hint"
        "-//tflite/python:schema_util"
        "-//tflite/python:test_util"
        "-//tflite/python:test_util_test"
        "-//tflite/python:tflite_keras_util"
        "-//tflite/python:util"
        "-//tflite/python/metrics:metrics_test"
        "-//tflite/python/metrics:metrics_wrapper"
        "-//tflite/python/optimize:calibrator"
        "-//tflite/python/optimize:calibrator_test"
        "-//tflite/python/testdata:double_op_and_kernels"
        "-//tflite/schema:upgrade_schema"
        "-//tflite/schema:upgrade_schema_main_lib"
        "-//tflite/tools:convert_image_to_csv"
        "-//tflite/tools:convert_image_to_csv_lib"
        "-//tflite/tools:flatbuffer_utils"
        "-//tflite/tools:flatbuffer_utils_test"
        "-//tflite/tools:randomize_weights"
        "-//tflite/tools:reverse_xxd_dump_from_cc"
        "-//tflite/tools:strip_strings"
        "-//tflite/tools:visualize_test"
        "-//tflite/tools/optimize:quantization_utils_test"
        "-//tflite/tools/optimize:quantize_model_test"
        "-//tflite/tools/optimize:reduced_precision_support_test"
        "-//tflite/tools/optimize/calibration:calibrator_test"
        "-//tflite/tools/optimize/python:modify_model_interface"
        "-//tflite/tools/optimize/python:modify_model_interface_constants"
        "-//tflite/tools/optimize/python:modify_model_interface_lib"
        "-//tflite/tools/optimize/python:modify_model_interface_lib_test"
        "-//tflite/tools/serialization:writer_lib_test"
        "-//tflite/tools/versioning:gpu_compatibility_test"
        "-//tflite/tools/versioning:op_signature_test"
)

LITERT_EXCLUDED_TARGETS=(
        "-//litert/c:litert_compiled_model_shared_lib_test"
        "-//litert/c:litert_compiled_model_test"
        "-//litert/cc:litert_compiled_model_test"
        "-//litert/python/tools/model_utils/test/..."
        # Requires mGPU environment.
        "-//litert/cc:litert_environment_test"
        "-//litert/runtime:compiled_model_test"
        "-//litert/runtime/accelerators/gpu/..."
        # Requires c++20.
        "-//litert/tools:tool_display_test"
        # Requires c++20.
        "-//litert/tools:dump_test"
        # Requires c++20.
        "-//litert/tools:apply_plugin_test"
)

# //litert targets that need the real TensorFlow (litert/compiler, the
# converter pywrap, ...). They are not dropped from CI:
# - With LITERT_WITH_TENSORFLOW=1 (Internal CI 'cpu_full', Linux), they are
#   built with --config=with_tensorflow. They are outside //tflite/..., so they
#   are added to the //tflite/... test run as LITERT_TF_TARGETS below.
# - Without it, they are added to LITERT_EXCLUDED_TARGETS because the default
#   build has no TensorFlow.
LITERT_TF_EXCLUDED_TARGETS=(
        "-//litert/compiler/..."
        "-//litert/python/mlir/..."
        "-//litert/python/tools/model_utils/..."
        "-//litert/integration_test/models:single_op"
        "-//litert/integration_test/models:single_op_files"
        "-//litert/python:_pywrap_litert_with_converter_0_pywrap"
        "-//litert/python:_pywrap_litert_with_converter_0_shared_object"
        "-//litert/python:_pywrap_litert_with_converter_1_pywrap"
        "-//litert/python:_pywrap_litert_with_converter_1_shared_object"
        "-//litert/python:_pywrap_litert_with_converter_2_pywrap"
        "-//litert/python:_pywrap_litert_with_converter_2_shared_object"
        "-//litert/python:_pywrap_litert_with_converter_3_pywrap"
        "-//litert/python:_pywrap_litert_with_converter_3_shared_object"
        "-//litert/python:_pywrap_litert_with_converter_common_split"
        "-//litert/python:_pywrap_litert_with_converter_info_collector"
        "-//litert/python:_pywrap_litert_with_converter_linker_input_filters"
        "-//litert/python:libpywrap_litert_with_converter_common.dylib"
        "-//litert/python:libpywrap_litert_with_converter_common.so"
        "-//litert/python:pywrap_litert_with_converter"
        "-//litert/python:pywrap_litert_with_converter_all_binaries"
        "-//litert/python:pywrap_litert_with_converter_binaries"
        "-//litert/python:pywrap_litert_with_converter_binaries.json"
        "-//litert/python:pywrap_litert_with_converter_common"
        "-//litert/python:pywrap_litert_with_converter_common.dll"
        "-//litert/python:pywrap_litert_with_converter_common_binaries"
        "-//litert/python:pywrap_litert_with_converter_common_cc_library"
        "-//litert/python:pywrap_litert_with_converter_common_if_lib"
        "-//litert/python:pywrap_litert_with_converter_common_import"
)

# Flags without --config=with_tensorflow, to match GitHub Actions CI.
SHIM_BUILD_FLAGS=("${COMMON_BUILD_FLAGS[@]}")

# Targets that need TensorFlow are only tested here, in Internal CI 'cpu_full'
# on Linux. GitHub Actions CI (Linux, macOS and Windows) does not test them.
LITERT_TF_TARGETS=()
if [ "$LITERT_WITH_TENSORFLOW" == "1" ]; then
    COMMON_BUILD_FLAGS+=("--config=with_tensorflow")
    for target in "${LITERT_TF_EXCLUDED_TARGETS[@]}"; do
        LITERT_TF_TARGETS+=("${target#-}")
    done
    LITERT_TF_TARGETS+=("${LITERT_EXCLUDED_TARGETS[@]}")
else
    EXCLUDED_TARGETS+=("${TF_EXCLUDED_TARGETS[@]}")
    LITERT_EXCLUDED_TARGETS+=("${LITERT_TF_EXCLUDED_TARGETS[@]}")
fi

if [ "$LITERT_TARGETS_ONLY" == "true" ]; then
    bazel test "${COMMON_BUILD_FLAGS[@]}" "${TEST_FLAGS[@]}" -- //litert/... "${LITERT_EXCLUDED_TARGETS[@]}"
else
    # Build core TFLite targets to populate remote cache for presubmits (e.g. tflite_bazel_cmake.yml).
    # LINT.IfChange(tflite_bazel_targets)
    bazel build \
      "${SHIM_BUILD_FLAGS[@]}" \
      -- \
      //tflite:tensorflowlite \
      //tflite/c:tensorflowlite_c \
      //tflite/tools/benchmark:benchmark_model \
      //tflite/converter:flatbuffer_translate
    # LINT.ThenChange(../workflows/tflite_bazel_cmake.yml:tflite_bazel_targets)
    bazel test "${COMMON_BUILD_FLAGS[@]}" "${TEST_FLAGS[@]}" -- //tflite/... "${EXCLUDED_TARGETS[@]}" \
      "${LITERT_TF_TARGETS[@]}"
fi
