# Copyright 2026 Google LLC.
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

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # "//third_party/py/litert_torch:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "gated_delta_update_impl",
    srcs = ["gated_delta_update_impl.cc"],
    hdrs = ["gated_delta_update_impl.h"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "gated_delta_update_litert_custom_op",
    srcs = ["gated_delta_update_litert_custom_op.cc"],
    hdrs = ["gated_delta_update_litert_custom_op.h"],
    deps = [
        ":gated_delta_update_impl",
        "//litert/cc:litert_common",
        "//litert/cc:litert_custom_op_kernel",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "@flatbuffers",
    ],
)

cc_test(
    name = "gated_delta_update_litert_custom_op_test",
    srcs = ["gated_delta_update_litert_custom_op_test.cc"],
    deps = [
        ":gated_delta_update_litert_custom_op",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_options",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/test:matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gated_delta_update_tflite_op",
    srcs = ["gated_delta_update_tflite_op.cc"],
    hdrs = ["gated_delta_update_tflite_op.h"],
    deps = [
        ":gated_delta_update_impl",
        "//tflite:mutable_op_resolver",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor_ctypes",
    ],
)

pybind_extension(
    name = "gated_delta_rule_kernels",
    srcs = ["gated_delta_rule_kernels.cc"],
    deps = [
        ":gated_delta_update_tflite_op",
        "//tflite/c:common",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)
