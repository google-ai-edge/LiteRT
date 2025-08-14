# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "log",
    srcs = select({
        "@org_tensorflow//tensorflow:android": ["log_android.cc"],
        "//conditions:default": ["log_default.cc"],
    }),
    hdrs = ["log.h"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/vendors/qualcomm/core:common",
    ],
)

cc_library(
    name = "miscs",
    srcs = ["miscs.cc"],
    hdrs = ["miscs.h"],
    deps = [
        ":log",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    deps = [
        ":log",
        ":miscs",
        "//litert/vendors/qualcomm/core:common",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_exec(
    name = "utils_device_test",
    backend_id = "qualcomm",
    target = ":utils_test",
)

cc_library(
    name = "qnn_model",
    srcs = ["qnn_model.cc"],
    hdrs = ["qnn_model.h"],
    deps = [
        ":log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)
