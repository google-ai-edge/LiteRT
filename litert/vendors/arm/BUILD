#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    default_visibility = ["//litert:__subpackages__"],
)

build_test(
    name = "build_so_test",
    targets = [
        "//litert/vendors/arm/compiler:compiler_plugin_so",
        "//litert/vendors/arm/dispatch:dispatch_api_so",
    ],
)
