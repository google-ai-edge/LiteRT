# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "htp_perf_control",
    srcs = ["htp_perf_control.cc"],
    hdrs = ["htp_perf_control.h"],
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/base:core_headers",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_backend",
    srcs = ["qnn_backend.cc"],
    hdrs = ["qnn_backend.h"],
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "ir_backend",
    srcs = ["ir_backend.cc"],
    hdrs = ["ir_backend.h"],
    deps = [
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "htp_backend",
    srcs = ["htp_backend.cc"],
    hdrs = ["htp_backend.h"],
    deps = [
        ":htp_perf_control",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "qnn_backend_test",
    srcs = [
        "qnn_backend_test.cc",
    ],
    deps = [
        ":htp_backend",
        ":ir_backend",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_googletest//:gtest_main",
    ],
)
