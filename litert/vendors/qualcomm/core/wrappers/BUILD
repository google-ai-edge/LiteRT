# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
)

cc_library(
    name = "quantize_params_wrapper",
    srcs = ["quantize_params_wrapper.cc"],
    hdrs = ["quantize_params_wrapper.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "tensor_wrapper",
    srcs = ["tensor_wrapper.cc"],
    hdrs = ["tensor_wrapper.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "param_wrapper",
    srcs = ["param_wrapper.cc"],
    hdrs = ["param_wrapper.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":tensor_wrapper",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "op_wrapper",
    srcs = ["op_wrapper.cc"],
    hdrs = ["op_wrapper.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":param_wrapper",
        ":tensor_wrapper",
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@qairt//:qnn_lib_headers",
    ],
)
