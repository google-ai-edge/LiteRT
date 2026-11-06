# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

proto_library(
    name = "evaluation_proto",
    srcs = ["evaluation.proto"],
)

cc_proto_library(
    name = "evaluation_cc_proto",
    deps = [":evaluation_proto"],
)

cc_library(
    name = "data",
    srcs = [
        "data.cc",
        "image.cc",
    ],
    hdrs = [
        "data.h",
        "image.h",
    ],
    copts = ["-fno-exceptions"],
    deps = [
        ":evaluation_cc_proto",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@stblib//:stb_image_hdrs",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_test(
    name = "image_test",
    size = "small",
    srcs = ["image_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/c:litert_model_types",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:types",
        "@XNNPACK",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/cc:litert_expected",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":metrics",
        "@com_google_googletest//:gtest_main",
    ],
)
