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
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_deps", "litert_gpu_accelerator_prebuilts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "accuracy_debugger_util",
    srcs = ["accuracy_debugger_util.cc"],
    hdrs = ["accuracy_debugger_util.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/options:litert_gpu_options",
        "//litert/compiler/plugin:algo",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools",
        "//litert/tools:dump_ops_util",
        "//litert/tools:tensor_utils",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "accuracy_debugger",
    srcs = ["accuracy_debugger_main.cc"],
    data = litert_gpu_accelerator_prebuilts(),
    deps = [
        ":accuracy_debugger_util",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_gpu_options",
        "//litert/core/model:model_load",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags/vendors:google_tensor_flags",
        "//litert/tools/flags/vendors:intel_openvino_flags",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + litert_gpu_accelerator_deps(),
)

cc_test(
    name = "accuracy_debugger_util_test",
    srcs = ["accuracy_debugger_util_test.cc"],
    data = ["//litert/test:testdata/simple_add_op.tflite"],
    deps = [
        ":accuracy_debugger_util",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_options",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/core/model",
        "//litert/test:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
