# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_test(
    name = "add_relu_test",
    srcs = [
        "add_relu_test.cc",
    ],
    deps = [
        "//litert/cc:litert_buffer_ref",
        "//litert/c:litert_model",
        "//litert/test:matchers",
        "//litert/test/generators:graph_helpers",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:qnn_backend_creator",
        "//litert/vendors/qualcomm/qnn_backend_test:utils",
        "@com_google_googletest//:gtest_main",
    ],
)
