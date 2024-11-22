# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Utils package for the stable delegate APIs.
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//tflite:build_def.bzl", "tflite_cc_shared_object", "tflite_linkopts_no_undefined")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

cc_library_with_tflite(
    name = "delegate_loader",
    srcs = ["delegate_loader.cc"],
    hdrs = ["delegate_loader.h"],
    generate_opaque_delegate_target = True,
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    tflite_deps = ["//tflite/acceleration/configuration/c:stable_delegate"],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/experimental/acceleration/compatibility:android_info",
        "//tflite/tools:logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_with_tflite(
    name = "stable_delegate_interface",
    hdrs = ["stable_delegate_interface.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = ["//tflite/acceleration/configuration/c:stable_delegate"],
    visibility = ["//visibility:public"],
)

tflite_cc_shared_object(
    name = "tensorflowlite_stable_xnnpack_delegate",
    testonly = True,
    srcs = ["stable_xnnpack_delegate.cc"],
    defines = ["TFL_STABLE_DELEGATE_COMPILE_LIBRARY"],
    linkopts = tflite_linkopts_no_undefined() + select({
        "@org_tensorflow//tensorflow:windows": [],
        # iOS linker doesn't support -version-script.
        "@org_tensorflow//tensorflow:ios": [
            # TODO(b/260213307): support hiding unnecessary symbols on iOS.
        ],
        "//conditions:default": [
            # Expose necessary symbols only.
            "-Wl,--version-script,$(location //tflite/delegates/utils/experimental/stable_delegate:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    visibility = ["//visibility:public"],
    deps = [
        ":stable_delegate_interface",
        ":version_script.lds",
        "//tflite/acceleration/configuration/c:stable_delegate",
        "//tflite/acceleration/configuration/c:xnnpack_plugin",
        "//tflite/c:c_api",
    ],
)

cc_test(
    name = "delegate_loader_test",
    srcs = ["delegate_loader_test.cc"],
    data = ["//tflite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate"],
    deps = [
        ":delegate_loader",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/acceleration/configuration/c:stable_delegate",
        "//tflite/delegates/utils/experimental/sample_stable_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_settings_json_parser",
    srcs = [
        "tflite_settings_json_parser.cc",
        "//tflite/acceleration/configuration:configuration_fbs_contents-inl.h",
    ],
    hdrs = ["tflite_settings_json_parser.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/kernels/internal:compatibility",
        "//tflite/tools:logging",
        "@flatbuffers",
    ],
)

build_test(
    name = "stable_delegate_build_test",
    targets = [
        ":delegate_loader",
        ":test_xnnpack_settings.json",
        ":tensorflowlite_stable_xnnpack_delegate",
        ":tflite_settings_json_parser",
    ],
)

cc_test(
    name = "tflite_settings_json_parser_test",
    srcs = ["tflite_settings_json_parser_test.cc"],
    data = [
        ":test_xnnpack_settings.json",
        "//tflite/tools/delegates/experimental/stable_delegate:test_invalid_settings.json",
    ],
    deps = [
        ":tflite_settings_json_parser",
        "//tflite/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "kernel_test_main",
    testonly = 1,
    srcs = ["kernel_test_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/c:c_api_types",
        "//tflite/kernels:acceleration_test_util",
        "//tflite/kernels:acceleration_test_util_internal",
        "//tflite/kernels:test_delegate_providers_lib",
        "//tflite/kernels:test_util",
        "//tflite/kernels:test_util_delegate_providers",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "stable_delegate_test_suite",
    size = "medium",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":kernel_test_main",
        "//tflite/kernels:combined_all_kernel_tests_lib",
    ],
)

exports_files(
    srcs = [
        "test_xnnpack_settings.json",
        "version_script.lds",
    ],
    visibility = ["//visibility:public"],
)
