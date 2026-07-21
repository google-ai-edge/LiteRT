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

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_native_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tflite/converter:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":extra_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
        "cc",
        "json",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":importer_test_min_max",
        ":test_schema.fbs",
        "//tflite/converter:flatbuffer_to_string",
        "//tflite/converter:flatbuffer_translate",
        "//tflite/converter:json_to_flatbuffer",
        "//tflite/converter:litert-opt",
        "//tflite/converter:tf_tfl_translate",
        "@llvm-project//llvm:FileCheck",
    ],
)

# Bundle together all the files that are used by the non-mlir file-based tests.
filegroup(
    name = "extra_files",
    srcs = glob(
        [
            "**/importer_test_min_max.cc.mlir",
            "**/reshape.mlir",
        ],
    ),
)

# A binary to inject min/max to a tflite model.
# A file check command is used to verify the imported result from this
# binary format.
tf_native_cc_binary(
    name = "importer_test_min_max",
    srcs = [
        "importer_test_min_max.cc",
    ],
    deps = [
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)
