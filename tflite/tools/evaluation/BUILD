# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
load("//tflite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_stable_tflite_abi", "cc_test_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/**",
]))

cc_library(
    name = "evaluation_stage",
    hdrs = ["evaluation_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//tflite/core/c:common",
        "//tflite/tools/evaluation/proto:evaluation_config_cc_proto",
    ],
)

cc_library_with_stable_tflite_abi(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    non_stable_abi_deps = [
        "//tflite/delegates/nnapi:nnapi_delegate",
    ] + select({
        "@org_tensorflow//tensorflow:ios": [
            "//tflite/delegates/coreml:coreml_delegate",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "//tflite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//tflite/delegates/gpu:supports_gpu_delegate": [
            "//tflite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "@org_tensorflow//tensorflow:arm_any": [
            "//tflite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    tflite_deps = [
        "//tflite/c:c_api",
        "//tflite/c:common",
        "//tflite/acceleration/configuration/c:delegate_plugin",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "//tflite/tools/delegates:delegate_provider_lib",
    ],
    tflite_deps_selects = [{
        "//tflite:tflite_with_xnnpack_explicit_false": [],
        "//conditions:default": [
            "//tflite/acceleration/configuration/c:xnnpack_plugin",
        ],
    }],
    deps = [
        "//tflite/tools:logging",
        "@flatbuffers",
    ] + select({
        "//tflite:tflite_with_xnnpack_explicit_false": [],
        "//conditions:default": [
            "//tflite/acceleration/configuration:configuration_fbs",
            "//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        ],
    }),
)

cc_library(
    name = "evaluation_delegate_provider",
    srcs = ["evaluation_delegate_provider.cc"],
    hdrs = ["evaluation_delegate_provider.h"],
    copts = tflite_copts(),
    deps = [
        ":utils",
        "//tflite/c:c_api_types",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools:tool_params",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "//tflite/tools/delegates:tflite_execution_providers",
        "//tflite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)

cc_test_with_tflite(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = [
        "testdata/empty.txt",
        "testdata/labels.txt",
    ],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    tflite_deps = [
        ":utils",
    ],
    deps = [
        "//tflite:context",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "evaluation_delegate_provider_test",
    srcs = ["evaluation_delegate_provider_test.cc"],
    linkopts = tflite_linkopts(),
    deps = [
        ":evaluation_delegate_provider",
        "//tflite/tools:tool_params",
        "//tflite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
