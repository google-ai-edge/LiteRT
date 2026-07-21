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

# Provides stable ABI delegate.
load("//tflite:build_def.bzl", "tflite_copts")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

cc_library(
    name = "delegate_provider",
    srcs = [
        "stable_delegate_provider.cc",
    ],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/c:common",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools:tool_params",
        "//tflite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Stable delegate does not support Windows because the shared library loader hasn't been
        # extended to support Windows.
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "//tflite/acceleration/configuration:configuration_fbs",
            "//tflite/acceleration/configuration/c:delegate_plugin",
            "//tflite/acceleration/configuration/c:stable_delegate",
            "//tflite/delegates/utils/experimental/stable_delegate:delegate_loader",
            "//tflite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        ],
    }),
    # Statically registers itself with DelegateProviderRegistrar.
    alwayslink = 1,
)

cc_test(
    name = "delegate_provider_test",
    size = "small",
    srcs = ["stable_delegate_provider_test.cc"],
    data = [
        ":test_invalid_settings.json",
        ":test_missing_delegate_path_settings.json",
        ":test_missing_stable_delegate_settings.json",
        ":test_sample_stable_delegate_settings.json",
        ":test_stable_xnnpack_settings.json",
        "//tflite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate",
        "//tflite/delegates/utils/experimental/stable_delegate:tensorflowlite_stable_xnnpack_delegate",
    ],
    # Disable the test on Windows as the shared library loader doesn't support it.
    tags = ["no-windows"],
    deps = [
        ":delegate_provider",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/tools:tool_params",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "//tflite/tools/delegates:delegate_provider_lib",
        "@com_google_googletest//:gtest_main",
        "@pthreadpool",
    ],
)

exports_files(
    srcs = [
        "test_invalid_settings.json",
        "test_sample_stable_delegate_settings.json",
    ],
    visibility = ["//visibility:public"],
)
