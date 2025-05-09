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
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:concatenation_op_builder",
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//litert/vendors/qualcomm/core/builders:matmul_op_builder",
        "//litert/vendors/qualcomm/core/builders:op_builder",
        "//litert/vendors/qualcomm/core/builders:reshape_op_builder",
        "//litert/vendors/qualcomm/core/builders:slice_op_builder",
        "//litert/vendors/qualcomm/core/builders:softmax_op_builder",
        "//litert/vendors/qualcomm/core/builders:split_op_builder",
        "//litert/vendors/qualcomm/core/builders:transpose_op_builder",
        "//litert/vendors/qualcomm/core/builders:quantize_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
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
        ":matmul_convert",
        ":mha_to_sha",
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core:tensor_pool",
    ],
)

cc_library(
    name = "matmul_convert",
    srcs = ["matmul_convert.cc"],
    hdrs = ["matmul_convert.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core:tensor_pool",
    ],
)

cc_library(
    name = "mha_to_sha",
    srcs = ["mha_to_sha.cc"],
    hdrs = ["mha_to_sha.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/builders:concatenation_op_builder",
        "//litert/vendors/qualcomm/core/builders:op_builder",
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//litert/vendors/qualcomm/core/builders:matmul_op_builder",
        "//litert/vendors/qualcomm/core/builders:reshape_op_builder",
        "//litert/vendors/qualcomm/core/builders:slice_op_builder",
        "//litert/vendors/qualcomm/core/builders:split_op_builder",
        "//litert/vendors/qualcomm/core/builders:transpose_op_builder",
        "//litert/vendors/qualcomm/core/builders:softmax_op_builder",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core:tensor_pool",
    ],
)
