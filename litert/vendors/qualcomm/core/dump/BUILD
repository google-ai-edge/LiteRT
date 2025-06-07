# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "dump_graph",
    srcs = ["dump_graph.cc"],
    hdrs = ["dump_graph.h"],
    deps = [
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/container:flat_hash_set",
        "@qairt//:qnn_lib_headers",
    ]
)
