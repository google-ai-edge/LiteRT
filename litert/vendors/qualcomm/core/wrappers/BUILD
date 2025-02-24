# Copyright (c) Qualcomm Innovation Center, Inc.
# All Rights Reserved.

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "quantize_params_wrapper",
    srcs = ["quantize_params_wrapper.cc"],
    hdrs = ["quantize_params_wrapper.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//third_party/absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

cc_library(
    name = "tensor_wrapper",
    srcs = ["tensor_wrapper.cc"],
    hdrs = ["tensor_wrapper.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

cc_library(
    name = "param_wrapper",
    srcs = ["param_wrapper.cc"],
    hdrs = ["param_wrapper.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

cc_library(
    name = "op_wrapper",
    srcs = ["op_wrapper.cc"],
    hdrs = ["op_wrapper.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/wrappers:param_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)
