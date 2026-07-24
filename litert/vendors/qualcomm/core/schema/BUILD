# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "soc_table",
    hdrs = ["soc_table.h"],
    deps = [
        "//litert/vendors/qualcomm/core/utils:log",
    ],
)

cc_test(
    name = "soc_table_test",
    srcs = ["soc_table_test.cc"],
    deps = [
        ":soc_table",
        "@com_google_googletest//:gtest_main",
    ],
)
