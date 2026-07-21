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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps")

package(
    default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "build_custom_npu_model_lib",
    srcs = ["build_custom_npu_model.cc"],
    hdrs = ["build_custom_npu_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:dispatch_op_schema",
        "//litert/core/model",
        "//litert/core/model:model_serialize",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "build_custom_npu_model_main",
    srcs = ["build_custom_npu_model_main.cc"],
    deps = [
        ":build_custom_npu_model_lib",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "build_custom_npu_model_test",
    srcs = ["build_custom_npu_model_test.cc"],
    deps = [
        ":build_custom_npu_model_lib",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc/internal:litert_extended_model",
        "//litert/core:dispatch_op_schema",
    ] + gtest_main_no_heapcheck_deps(),
)
