#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "capabilities",
    srcs = ["capabilities.cc"],
    hdrs = ["capabilities.h"],
    deps = [
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
    ],
)

build_test(
    name = "build_so_test",
    targets = [
        "//litert/vendors/arm/compiler:compiler_plugin_so",
        "//litert/vendors/arm/dispatch:dispatch_api_so",
    ],
)
