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

load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
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
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)
