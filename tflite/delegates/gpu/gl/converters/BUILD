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

load(
    "@org_tensorflow//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bhwc_to_phwc4",
    srcs = ["bhwc_to_phwc4.cc"],
    hdrs = ["bhwc_to_phwc4.h"],
    deps = [
        ":util",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/gl:command_queue",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:gl_program",
        "//tflite/delegates/gpu/gl:gl_shader",
        "//tflite/delegates/gpu/gl:variable",
    ],
)

cc_test(
    name = "bhwc_to_phwc4_test",
    size = "small",
    srcs = ["bhwc_to_phwc4_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":bhwc_to_phwc4",
        "//tflite/delegates/gpu/common:convert",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "phwc4_to_bhwc",
    srcs = ["phwc4_to_bhwc.cc"],
    hdrs = ["phwc4_to_bhwc.h"],
    deps = [
        ":util",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/gl:command_queue",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:gl_program",
        "//tflite/delegates/gpu/gl:gl_shader",
        "//tflite/delegates/gpu/gl:variable",
    ],
)

cc_test(
    name = "phwc4_to_bhwc_test",
    size = "small",
    srcs = ["phwc4_to_bhwc_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":phwc4_to_bhwc",
        "//tflite/delegates/gpu/common:convert",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
