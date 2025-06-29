# Copyright 2024 Google LLC.
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

load("//litert/build_common:tfl_model_gen.bzl", "tfl_model_gen")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_public",
    ],
)

tfl_model_gen(
    name = "mlir_test_data",
    srcs = glob(["testdata/*.mlir"]),
)

filegroup(
    name = "tflite_test_data",
    srcs = glob(["testdata/*.tflite"]),
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*.tflite",
        "testdata/*.bin",
    ]) + [":mlir_test_data"],
)

cc_library(
    name = "common",
    testonly = 1,
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/core:filesystem",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "simple_model",
    testonly = 1,
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_model.tflite",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_model_custom_op",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_model_custom_op_test_vectors.h",
    ],
    data = [
        "testdata/simple_model_custom_op.tflite",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_add_uint16_qualcomm.bin",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
        "testdata/simple_model_npu.tflite",
        "testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_cascade_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_cascade_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_cascade_model_npu.tflite",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_mixed_cascade_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_mixed_cascade_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_mixed_cascade_model_npu.tflite",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "test_models",
    hdrs = ["test_models.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "rng_fixture",
    testonly = True,
    hdrs = ["rng_fixture.h"],
    deps = [
        ":fuzz",
        "//litert/cc:litert_rng",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "matchers_test",
    srcs = ["matchers_test.cc"],
    deps = [
        ":matchers",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_googletest//:gtest_main",
    ],
)

# Use this library if you want to enforce an OSS environment for your test.
cc_library(
    name = "matchers_oss",
    testonly = True,
    hdrs = ["matchers.h"],
    defines = ["LITERT_DEFINE_GTEST_STATUS_PRINTER"],
    tags = ["avoid_dep"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "fuzz",
    hdrs = ["fuzz.h"],
)

exports_files(srcs = [
    "testdata/mobilenet_v2_1.0_224.tflite",
    "testdata/shared_input_cpu_npu_google_tensor_precompiled.tflite",
    "testdata/simple_add_op_google_tensor_p25_precompiled.tflite",
    "testdata/simple_add_op_qc_v75_precompiled.tflite",
    "testdata/simple_add_uint16_qualcomm.bin",
    "testdata/simple_model_google_tensor.bin",
    "testdata/simple_model_mtk.bin",
    "testdata/simple_model_npu_google_tensor_precompiled.tflite",
    "testdata/simple_model_qualcomm.bin",
    "testdata/simple_model_qualcomm_sm8650_precompiled.tflite",
    "testdata/simple_quantized_ops.tflite",
])

# Memory safety test for unowned buffer API
cc_binary(
    name = "unowned_buffer_memory_test",
    testonly = 1,
    srcs = ["unowned_buffer_memory_test.cc"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_model",
    ],
    # Can be run with: bazel run --config=asan //litert/test:unowned_buffer_memory_test
)

# Verification test to demonstrate no-copy behavior
cc_binary(
    name = "verify_no_copy_test",
    testonly = 1,
    srcs = ["verify_no_copy_test.cc"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model",
    ],
)
