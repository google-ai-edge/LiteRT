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

# Valid values for PLATFORM_NAME are "ubuntu", "android", "ios".
PLATFORM_NAME="${PLATFORM_NAME:-ubuntu}"

UBUNTU_BUILD_FLAGS=("-c" "opt"
    "--cxxopt=--std=c++17"
    "--copt=-Wno-gnu-offsetof-extensions"
    "--build_tag_filters=-no_oss,-oss_serial,-gpu,-tpu,-v1only"
    # Re-enable the following when the compiler supports AVX_VNNI
    "--define=xnn_enable_avxvnni=false"
    # Re-enable the following when the compiler supports AVX512-AMX
    "--define=xnn_enable_avx512amx=false"
    # Re-enable the foolowing when the compiler supports AVX512_FP16 (clang > 15,
    # GCC > 13)
    "--define=xnn_enable_avx512fp16=false"
    "--nocheck_visibility"
    "--show_timestamps"
  )

ANDROID_BUILD_FLAGS=(
  # Enable optimization because it sometimes exposes compiler bugs.
  "--compilation_mode=opt"
  # Use "lite" protos to reduce binary size.
  "--config=force_compact_protos"
  "--config=android_arm"
  "--android_ndk_min_sdk_version=28"
  "--show_timestamps"
  )

IOS_BUILD_FLAGS=(
  "--config=ios_x86_64"
  "--compilation_mode=opt"
  "--swiftcopt=-enable-testing"
  # TODO(b/287670077): remove once code has been updated to handle newer versions of xcode
  "--xcode_version=15.4.0"
  "--show_timestamps"
  )

# TODO(b/383171538): Remove the following excluded targets once their bazel build are fixed.
UBUNTU_EXCLUDED_TARGETS=(
  "-//tflite:tflite_internal_cc_3p_api_deps_src_all"
  "-//tflite/delegates/coreml:coreml_delegate"
  "-//tflite/delegates/coreml:coreml_delegate_kernel"
  "-//tflite/delegates/coreml:coreml_executor"
  "-//tflite/delegates/flex:libtensorflowlite_flex.dylib"
  "-//tflite/delegates/flex:libtensorflowlite_flex.so"
  "-//tflite/delegates/flex:tensorflowlite_flex"
  "-//tflite/delegates/flex:tensorflowlite_flex.dll"
  "-//tflite/delegates/flex/test:TestTensorFlowLiteSelectTfOps_framework"
  "-//tflite/delegates/flex/test:framework_build_test"
  "-//tflite/delegates/flex/test:framework_build_test_0__deps"
  "-//tflite/delegates/flex/test:test_tensorflowlitelib_flex"
  "-//tflite/delegates/gpu:api"
  "-//tflite/delegates/gpu:async_buffers"
  "-//tflite/delegates/gpu:delegate"
  "-//tflite/delegates/gpu:gl_delegate"
  "-//tflite/delegates/gpu:libtensorflowlite_gpu_delegate.so"
  "-//tflite/delegates/gpu:libtensorflowlite_gpu_gl.so"
  "-//tflite/delegates/gpu:metal_delegate"
  "-//tflite/delegates/gpu:metal_delegate_internal"
  "-//tflite/delegates/gpu:spi"
  "-//tflite/delegates/gpu:tensorflow_lite_gpu_dylib"
  "-//tflite/delegates/gpu:tensorflow_lite_gpu_framework"
  "-//tflite/delegates/gpu/cl:api"
  "-//tflite/delegates/gpu/cl:egl_sync"
  "-//tflite/delegates/gpu/cl:gl_interop"
  "-//tflite/delegates/gpu/cl:gpu_api_delegate"
  "-//tflite/delegates/gpu/cl:tensor_type_util"
  "-//tflite/delegates/gpu/cl/kernels:converter"
  "-//tflite/delegates/gpu/cl/testing:delegate_testing"
  "-//tflite/delegates/gpu/cl/testing:internal_api_samples"
  "-//tflite/delegates/gpu/cl/testing:performance_profiling"
  "-//tflite/delegates/gpu/common:model_builder_helper_test"
  "-//tflite/delegates/gpu/gl:api"
  "-//tflite/delegates/gpu/gl:api2"
  "-//tflite/delegates/gpu/gl:command_queue"
  "-//tflite/delegates/gpu/gl:egl_context"
  "-//tflite/delegates/gpu/gl:egl_environment"
  "-//tflite/delegates/gpu/gl:egl_surface"
  "-//tflite/delegates/gpu/gl:gl_buffer"
  "-//tflite/delegates/gpu/gl:gl_call"
  "-//tflite/delegates/gpu/gl:gl_errors"
  "-//tflite/delegates/gpu/gl:gl_program"
  "-//tflite/delegates/gpu/gl:gl_shader"
  "-//tflite/delegates/gpu/gl:gl_sync"
  "-//tflite/delegates/gpu/gl:gl_texture"
  "-//tflite/delegates/gpu/gl:gl_texture_helper"
  "-//tflite/delegates/gpu/gl:object_manager"
  "-//tflite/delegates/gpu/gl:request_gpu_info"
  "-//tflite/delegates/gpu/gl:runtime"
  "-//tflite/delegates/gpu/gl/compiler:compiled_node_test"
  "-//tflite/delegates/gpu/gl/converters:bhwc_to_phwc4"
  "-//tflite/delegates/gpu/gl/converters:phwc4_to_bhwc"
  "-//tflite/delegates/gpu/gl/kernels:converter"
  "-//tflite/delegates/gpu/gl/kernels:test_util"
  "-//tflite/delegates/gpu/gl/runtime:shared_buffer"
  "-//tflite/delegates/gpu/java/src/main/native:compatibility_list_jni"
  "-//tflite/delegates/gpu/java/src/main/native:gpu_delegate_jni"
  "-//tflite/delegates/gpu/java/src/main/native:native"
  "-//tflite/delegates/gpu/metal:TestBinary"
  "-//tflite/delegates/gpu/metal:buffer"
  "-//tflite/delegates/gpu/metal:buffer_convert"
  "-//tflite/delegates/gpu/metal:buffer_test_lib"
  "-//tflite/delegates/gpu/metal:common"
  "-//tflite/delegates/gpu/metal:common_test_lib"
  "-//tflite/delegates/gpu/metal:common_tests_lib"
  "-//tflite/delegates/gpu/metal:compute_task"
  "-//tflite/delegates/gpu/metal:compute_task_cc"
  "-//tflite/delegates/gpu/metal:gpu_object"
  "-//tflite/delegates/gpu/metal:inference_context"
  "-//tflite/delegates/gpu/metal:inference_context_cc_fbs_srcs"
  "-//tflite/delegates/gpu/metal:metal_arguments"
  "-//tflite/delegates/gpu/metal:metal_device"
  "-//tflite/delegates/gpu/metal:metal_spatial_tensor"
  "-//tflite/delegates/gpu/metal:metal_spatial_tensor_test_lib"
  "-//tflite/delegates/gpu/metal/benchmarking:benchmark_lib"
  "-//tflite/delegates/gpu/metal/benchmarking:iOSBenchmark"
  "-//tflite/delegates/gpu/metal/kernels:add_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:cast_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:concat_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:conv_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:conv_weights_converter_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:cumsum_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:depthwise_conv_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:elementwise_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:fully_connected_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:gather_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:kernel_tests_lib"
  "-//tflite/delegates/gpu/metal/kernels:lstm_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:max_unpooling_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:mean_stddev_normalization_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:one_hot_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:padding_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:pooling_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:prelu_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:quantize_and_dequantize_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:reduce_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:relu_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:resampler_test"
  "-//tflite/delegates/gpu/metal/kernels:resampler_test.__internal__.__test_bundle"
  "-//tflite/delegates/gpu/metal/kernels:resampler_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:reshape_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:resize_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:select_v2_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:slice_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:softmax_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:space_to_depth_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:split_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:test_util"
  "-//tflite/delegates/gpu/metal/kernels:tile_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:transpose_conv_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:transpose_test_lib"
  "-//tflite/delegates/gpu/metal/kernels:winograd_test_lib"
  "-//tflite/delegates/hexagon/java:libtensorflowlite_hexagon_jni.so"
  "-//tflite/delegates/hexagon/java:tensorflow-lite-hexagon"
  "-//tflite/delegates/hexagon/java:tensorflow-lite-hexagon_dummy_app_for_so"
  "-//tflite/delegates/hexagon/java:tensorflowlite_hexagon"
  "-//tflite/delegates/hexagon/java:tensorflowlite_java_hexagon"
  "-//tflite/delegates/xnnpack:reduce_test"
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
  "-//tflite/kernels:pywrap_variable_ops"
  "-//tflite/kernels:pywrap_variable_ops.so"
  "-//tflite/kernels:pywrap_variable_ops_filegroup"
  "-//tflite/kernels:pywrap_variable_ops_pyd_copy"
  "-//tflite/kernels/gradient:pywrap_gradient_ops"
  "-//tflite/kernels/gradient:pywrap_gradient_ops.so"
  "-//tflite/kernels/gradient:pywrap_gradient_ops_filegroup"
  "-//tflite/kernels/gradient:pywrap_gradient_ops_pyd_copy"
  "-//tflite/kernels/perception:pywrap_perception_ops"
  "-//tflite/kernels/perception:pywrap_perception_ops.so"
  "-//tflite/kernels/perception:pywrap_perception_ops_filegroup"
  "-//tflite/kernels/perception:pywrap_perception_ops_pyd_copy"
  "-//tflite/profiling:profile_summarizer_test"
  "-//tflite/profiling:profile_summary_formatter_test"
  "-//tflite/profiling:signpost_profiler"
  "-//tflite/python/interpreter_wrapper:interpreter_wrapper_lib"
  "-//tflite/python/optimize:calibration_wrapper_lib"
  "-//tflite/python/testdata:gather_string"
  "-//tflite/python/testdata:gather_string_0d"
  "-//tflite/python/testdata:interpreter_test_data"
  "-//tflite/python/testdata:permute_float"
  "-//tflite/python/testdata:permute_uint8"
  "-//tflite/schema:tflite_internal_cc_3p_api_deps_src"
  "-//tflite/testing:customized_tflite_for_add_ops"
  "-//tflite/testing:customized_tflite_for_add_ops_aar"
  "-//tflite/testing:customized_tflite_for_add_ops_aar_dummy_app_for_so"
  "-//tflite/testing:customized_tflite_for_add_ops_jni"
  "-//tflite/testing:libtensorflowlite_jni.so"
  "-//tflite/testing:tflite_tflite_model_example_test_model"
  "-//tflite/core/async/testing:mock_async_kernel"
  "-//tflite/core/async:backend_async_kernel_interface"
  "-//tflite/tools/evaluation/proto/..."
  # The following are failing due to :flatbuffer_tools
  "-//tflite/c/..."
  "-//tflite/core/..."
  "-//tflite/experimental/litert/c/..."
  "-//tflite/experimental/litert/cc/..."
  "-//tflite/experimental/litert/core/..."
  "-//tflite/experimental/litert/runtime/..."
  "-//tflite/experimental/litert/test/..."
  "-//tflite/experimental/litert/tools/..."
  "-//tflite/experimental/litert/vendors/..."
  # The following below are android exclusive targets
  "-//tflite/acceleration/..."
  "-//tflite/delegates/gpu/gl:android_sync"
  "-//tflite/delegates/gpu/java/src/main/native/..."
  "-//tflite/delegates/hexagon/hexagon_nn:hexagon_interface_android"
  "-//tflite/delegates/hexagon/java/..."
  # For now remove unstable experimental code.
  "-//tflite/experimental/..."
  "-//tflite/java/..."
  "-//tflite/tools/benchmark/android/..."
  "-//tflite/tools/benchmark/experimental/delegate_performance/android/..."
  "-//tflite/tools/benchmark/experimental/firebase/android/..."
  # Note: dont need to exlude ios as ios starts with BAZEL.apple
)

ANDROID_TARGETS=(
  "//tflite/acceleration/..."
  "//tflite/delegates/gpu/gl:android_sync"
  "//tflite/delegates/gpu/java/src/main/native/..."
  "//tflite/delegates/hexagon/hexagon_nn:hexagon_interface_android"
  "//tflite/delegates/hexagon/java/..."
  "//tflite/experimental/acceleration/..."
  "//tflite/java/..."
  "//tflite/tools/benchmark/android/..."
  "//tflite/tools/benchmark/experimental/delegate_performance/android/..."
  "//tflite/tools/benchmark/experimental/firebase/android/..."
  # As described in b/141466757, we need to ensure benchmark tools are
  # buildable. To avoid creating new TAP targets, simply add benchmark
  # tools targets here.
  "//tflite/tools/benchmark:benchmark_model"
  "//tflite/tools/benchmark:benchmark_model_performance_options"
)

IOS_TARGETS=(
  "//tflite/ios/..."
  "//tflite/objc/..."
  "//tflite/swift/..."
  "//tflite/tools/benchmark/experimental/ios/..."
)

# Build targets for the specified platform.
if [ "$PLATFORM_NAME" == "ubuntu" ]; then
    bazel build "${UBUNTU_BUILD_FLAGS[@]}" -- //tflite/... //litert/... "${UBUNTU_EXCLUDED_TARGETS[@]}"
elif [ "$PLATFORM_NAME" == "android" ]; then
    bazel build "${ANDROID_BUILD_FLAGS[@]}" -- "${ANDROID_TARGETS[@]}"
elif [ "$PLATFORM_NAME" == "ios" ]; then
    bazel build "${IOS_BUILD_FLAGS[@]}" -- "${IOS_TARGETS[@]}"
else
    echo "Unsupported platform: ${PLATFORM_NAME}"
    exit 1
fi
