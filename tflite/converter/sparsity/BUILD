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

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        ":friends",
        "@org_tensorflow//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tflite/...",
        "//tflite/converter/...",
    ],
)

cc_library(
    name = "sparsify_model",
    srcs = [
        "sparsify_model.cc",
    ],
    hdrs = [
        "sparsify_model.h",
    ],
    deps = [
        "//tflite/converter:flatbuffer_translate_lib",
        "//tflite/converter:pass_registry_utils",
        "//tflite/converter:tensorflow_lite_d2s",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "sparsify_model_test",
    srcs = ["sparsify_model_test.cc"],
    data = [
        ":testdata/sparse_tensor.bin",
    ],
    deps = [
        ":sparsify_model",
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
