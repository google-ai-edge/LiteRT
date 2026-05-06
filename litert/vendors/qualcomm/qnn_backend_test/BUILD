# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "litert_test", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_test(
    name = "qnn_model_test",
    srcs = [
        "qnn_model_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "manual",
        # Don't build/test in OS until qnn is available.
        "no-remote-exec",
        "nobuilder",
        "notap",
    ],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":test_utils",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//litert/test:common",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/utils:test_utils",
        "@com_google_googletest//:gtest",
    ],
)
