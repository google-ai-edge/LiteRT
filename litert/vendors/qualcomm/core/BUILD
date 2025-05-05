# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "op_code",
    hdrs = ["op_code.h"],
)

cc_library(
    name = "tensor_pool",
    srcs = ["tensor_pool.cc"],
    hdrs = ["tensor_pool.h"],
    deps = [
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "tensor_pool_test",
    srcs = [
        "tensor_pool_test.cc",
    ],
    deps = [
        ":tensor_pool",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_exec(
    name = "tensor_pool_device_test",
    backend_id = "qualcomm",
    target = ":tensor_pool_test",
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
)
