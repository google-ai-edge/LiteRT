# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_test(
    name = "op_wrapper_test",
    srcs = [
        "op_wrapper_test.cc",
    ],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_test(
    name = "tensor_wrapper_test",
    srcs = [
        "tensor_wrapper_test.cc",
    ],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_test(
    name = "param_wrapper_test",
    srcs = [
        "param_wrapper_test.cc",
    ],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:param_wrapper",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_test(
    name = "quantize_params_wrapper_test",
    srcs = [
        "quantize_params_wrapper_test.cc",
    ],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
    ],
)
