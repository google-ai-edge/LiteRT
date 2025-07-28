# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/schema:soc_table",
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
        "//litert/vendors/qualcomm/core/utils:log",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "htp_backend",
    srcs = ["htp_backend.cc"],
    hdrs = ["htp_backend.h"],
    deps = [
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@qairt//:qnn_lib_headers",
    ],
)
