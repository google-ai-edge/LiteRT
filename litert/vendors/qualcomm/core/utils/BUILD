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
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core:common",
    ],
)

cc_library(
    name = "miscs",
    srcs = ["miscs.cc"],
    hdrs = ["miscs.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    deps = [
        ":log",
        ":miscs",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "//litert/vendors/qualcomm/core:common",
    ],
)

litert_device_exec(
    name = "utils_device_test",
    backend_id = "qualcomm",
    target = ":utils_test",
)
