# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert/vendors/qualcomm:__subpackages__"],
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
        "//tflite/experimental/litert/vendors/qualcomm/core:common",
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
        "@com_google_googletest//:gtest_main",
    ],
)
