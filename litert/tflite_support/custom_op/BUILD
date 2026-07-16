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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:litert_public",
    ],
    features = ["allow_starlark_transitions"],
)

cc_library(
    name = "custom_op",
    srcs = ["custom_op.cc"],
    hdrs = ["custom_op.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_options",
        "//litert/core:options",
    ],
)

cc_test(
    name = "custom_op_test",
    srcs = ["custom_op_test.cc"],
    data = [
        "//litert/test:testdata/custom_sinh.bin",
        "//litert/test:testdata/max_unpooling_2d.tflite",
        "//litert/test:testdata/max_unpooling_2d_valid.tflite",
    ],
    deps = [
        ":custom_op",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//tflite:builtin_ops",
        "//tflite/core/c:private_c_api_opaque_without_op_resolver",
        "//tflite/core/c:private_common",
        "//tflite/kernels/perception:perception_ops",  # buildcleaner: keep
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
