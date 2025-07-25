# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
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

cc_test(
    name = "qnn_model_test",
    srcs = [
        "qnn_model_test.cc",
    ],
    deps = [
        ":qnn_backend_creator",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
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
