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

load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")
load("//third_party/tensorflow:tensorflow.google.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
)

glob_lit_tests(
    name = "lit_tests",
    data = [":test_utilities"],
    default_tags = [
        "no_oss",
        "nosan",
        "noasan",
        "nomsan",
        "notsan",
    ],
    driver = "run_lit.sh",
    test_file_exts = ["mlir"],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//third_party/llvm/llvm-project/llvm:FileCheck",
        "//third_party/llvm/llvm-project/llvm:not",
        "//third_party/llvm/llvm-project/mlir:run_lit.sh",
        "//litert/tools:apply_plugin_main_for_test",
        "//litert/vendors/examples:example_plugin_so",
        "//third_party/tensorflow/compiler/mlir/lite:flatbuffer_to_string",
        "//third_party/tensorflow/compiler/mlir/lite:flatbuffer_translate",
    ],
    # This test can be run only on Linux.
    target_compatible_with = select({
        "//third_party/bazel_platforms/os:linux": [],
        "//conditions:default": ["//third_party/bazel_platforms:incompatible"],
    }),
)
