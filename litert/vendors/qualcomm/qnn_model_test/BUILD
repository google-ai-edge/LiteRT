# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_test(
    name = "qnn_model_test",
    srcs = [
        "qnn_model_test.cc",
    ],
    deps = [
        ":utils",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm:qnn_manager",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/compiler:graph_mapper",
    ],
)

