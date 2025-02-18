# Copyright (c) Qualcomm Innovation Center, Inc.
# All Rights Reserved.

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "tensor_pool",
    srcs = ["tensor_pool.cc"],
    hdrs = ["tensor_pool.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tflite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)
