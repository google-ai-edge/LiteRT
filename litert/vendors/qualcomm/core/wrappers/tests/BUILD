# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_test(
    name = "op_wrapper_test",
    srcs = [
        "op_wrapper_test.cc",
    ],
    deps = [
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_exec(
    name = "op_wrapper_device_test",
    backend_id = "qualcomm",
    target = ":op_wrapper_test",
)

cc_test(
    name = "tensor_wrapper_test",
    srcs = [
        "tensor_wrapper_test.cc",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_exec(
    name = "tensor_wrapper_device_test",
    backend_id = "qualcomm",
    target = ":tensor_wrapper_test",
)

cc_test(
    name = "param_wrapper_test",
    srcs = [
        "param_wrapper_test.cc",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/wrappers:param_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_exec(
    name = "param_wrapper_device_test",
    backend_id = "qualcomm",
    target = ":param_wrapper_test",
)

cc_test(
    name = "quantize_params_wrapper_test",
    srcs = [
        "quantize_params_wrapper_test.cc",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_exec(
    name = "quantize_params_wrapper_device_test",
    backend_id = "qualcomm",
    target = ":quantize_params_wrapper_test",
)
