# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "backend_utils",
    hdrs = ["backend_utils.h"],
    deps = [
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/types:span",
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
        ":backend_utils",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "dsp_backend",
    srcs = ["dsp_backend.cc"],
    hdrs = ["dsp_backend.h"],
    deps = [
        ":backend_utils",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "ir_backend_test",
    srcs = [
        "ir_backend_test.cc",
    ],
    deps = [
        ":ir_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "backend_utils_test",
    srcs = [
        "backend_utils_test.cc",
    ],
    deps = [
        ":backend_utils",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dsp_backend_test",
    srcs = [
        "dsp_backend_test.cc",
    ],
    deps = [
        ":backend_utils",
        ":dsp_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "htp_backend_test",
    srcs = [
        "htp_backend_test.cc",
    ],
    deps = [
        ":backend_utils",
        ":htp_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)
