# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_test(
    name = "qnn_backend_creator_test",
    srcs = [
        "qnn_backend_creator_test.cc",
    ],
    deps = [
        ":qnn_backend_creator",
        "//litert/vendors/qualcomm/core:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "qnn_backend_creator",
    srcs = ["qnn_backend_creator.cc"],
    hdrs = ["qnn_backend_creator.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//litert/cc:litert_model",
        "//litert/vendors/qualcomm/compiler:qnn_compose_graph",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)
