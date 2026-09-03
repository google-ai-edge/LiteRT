# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# TODO(b/120779485): Flesh out tests to exercise the following:
#   1) Inference with MobileNet against a fuzzed, valid input corpus.
cc_test(
    name = "simple_fuzzer_test",
    srcs = [
        "simple_fuzzer_test.cc",
    ],
    deps = [
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/kernels:test_util",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "model_fuzzer_test",
    srcs = [
        "model_fuzzer_test.cc",
    ],
    data = [
        "//tflite:testdata/add.bin",
    ],
    deps = [
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/core/tools:verifier",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)

cc_library(
    name = "fuzzing_support",
    srcs = [
        "fuzzer_quota_allocator.cc",
        "one_op_fuzz_model.cc",
    ],
    hdrs = [
        "fuzzer_quota_allocator.h",
        "fuzzing_util.h",
        "one_op_fuzz_model.h",
    ],
    deps = [
        "//tflite:framework",
        "//tflite:version",
        "//tflite/core:framework_stable",
        "//tflite/core/api:error_reporter",
        "//tflite/core/c:common",
        "//tflite/core/tools:verifier",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pad_fuzz_test",
    srcs = ["pad_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pad_xnnpack_fuzz_test",
    srcs = ["pad_fuzz_test.cc"],
    defines = ["TFLITE_PAD_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "mirror_pad_fuzz_test",
    srcs = ["mirror_pad_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "stablehlo_pad_fuzz_test",
    srcs = ["stablehlo_pad_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "convolution_fuzz_test",
    srcs = ["convolution_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "convolution_xnnpack_fuzz_test",
    srcs = ["convolution_fuzz_test.cc"],
    defines = ["TFLITE_CONVOLUTION_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "arg_min_max_fuzz_test",
    srcs = ["arg_min_max_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "cumsum_fuzz_test",
    srcs = ["cumsum_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "reduce_fuzz_test",
    srcs = ["reduce_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "reduce_xnnpack_fuzz_test",
    srcs = ["reduce_fuzz_test.cc"],
    defines = ["TFLITE_REDUCE_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "segment_fuzz_test",
    srcs = ["segment_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "topk_v2_fuzz_test",
    srcs = ["topk_v2_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "cast_fuzz_test",
    srcs = ["cast_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "reshape_fuzz_test",
    srcs = ["reshape_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:array",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels/internal:reshape_utils",
        "//tflite/kernels/internal:runtime_shape",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "reshape_xnnpack_fuzz_test",
    srcs = ["reshape_fuzz_test.cc"],
    defines = ["TFLITE_RESHAPE_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:array",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/kernels/internal:reshape_utils",
        "//tflite/kernels/internal:runtime_shape",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "slice_fuzz_test",
    srcs = ["slice_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "slice_xnnpack_fuzz_test",
    srcs = ["slice_fuzz_test.cc"],
    defines = ["TFLITE_SLICE_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
