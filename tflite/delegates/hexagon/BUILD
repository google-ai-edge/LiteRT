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

load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "@bazel_skylib//rules:common_settings.bzl",
    "bool_flag",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# If --define EXPLICIT_DISABLE_HEXAGON=1 is passed, the Hexagon delegate will be disabled.
config_setting(
    name = "explicit_disable_hexagon_define",
    define_values = {
        "EXPLICIT_DISABLE_HEXAGON": "1",
    },
)

bool_flag(
    name = "explicit_disable_hexagon",
    build_setting_default = False,
)

config_setting(
    name = "explicit_disable_hexagon_flag",
    flag_values = {
        ":explicit_disable_hexagon": "True",
    },
)

# Set if either the explicit_disable_hexagon define or the flag is set.
selects.config_setting_group(
    name = "explicit_disable_hexagon_setting",
    match_any = [
        ":explicit_disable_hexagon_define",
        ":explicit_disable_hexagon_flag",
    ],
)

cc_library(
    name = "hexagon_implementation",
    srcs = ["hexagon_implementation.cc"],
    hdrs = [
        "hexagon_implementation.h",
        "hexagon_nn_interface.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tflite:minimal_logging",
        "//tflite/delegates/hexagon/hexagon_nn:hexagon_nn_header",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "hexagon_delegate_kernel",
    srcs = [
        "hexagon_delegate.h",
        "hexagon_delegate_kernel.cc",
    ],
    hdrs = ["hexagon_delegate_kernel.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":hexagon_implementation",
        ":utils",
        "//tflite:kernel_api",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/delegates/hexagon/builders:op_builder",
        "//tflite/delegates/hexagon/hexagon_nn:hexagon_nn_header",
        "//tflite/delegates/utils:simple_delegate",
        "//tflite/schema:schema_fbs",
        "@hexagon_nn//:hexagon_nn_header",
        "@hexagon_nn//:hexagon_nn_ops",
    ],
)

cc_library(
    name = "enable_hexagon_delegate",
    defines = select({
        "@org_tensorflow//tensorflow:arm_any": ["TFLITE_ENABLE_HEXAGON"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "hexagon_delegate",
    srcs = select({
        ":explicit_disable_hexagon_setting": [],
        "//conditions:default": ["hexagon_delegate.cc"],
    }),
    hdrs = select({
        ":explicit_disable_hexagon_setting": [],
        "//conditions:default": ["hexagon_delegate.h"],
    }),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = select({
        ":explicit_disable_hexagon_setting": [],
        "//conditions:default": [
            ":hexagon_delegate_kernel",
            ":hexagon_implementation",
            ":utils",
            "//tflite:kernel_api",
            "//tflite:minimal_logging",
            "//tflite/core/c:common",
            "//tflite/delegates/utils:simple_delegate",
        ],
    }) + select({
        ":explicit_disable_hexagon_setting": [],
        "@org_tensorflow//tensorflow:ios": [],
        "@org_tensorflow//tensorflow:ios_x86_64": [],
        "@org_tensorflow//tensorflow:macos": [],
        "//conditions:default": [":enable_hexagon_delegate"],
    }),
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tflite:kernel_api",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkopts = tflite_linkopts() + ["-lm"],
    deps = [
        ":utils",
        "//tflite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files([
    "hexagon_delegate.h",
    "version_script.lds",
])
