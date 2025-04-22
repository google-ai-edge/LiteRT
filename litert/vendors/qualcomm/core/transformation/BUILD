# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_test(
    name = "graph_to_graph_test",
    srcs = [
        "graph_to_graph_test.cc",
    ],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":graph_to_graph",
        "@com_google_googletest//:gtest_main",
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:matmul_op_builder",
        "//litert/vendors/qualcomm/core/builders:quantize_op_builder",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "graph_to_graph",
    srcs = ["graph_to_graph.cc"],
    hdrs = ["graph_to_graph.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
    ],
)
