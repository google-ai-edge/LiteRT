load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite/delegates/gpu:build_defs.bzl", "gtest_main_no_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "conv_pointwise",
    srcs = ["conv_pointwise.cc"],
    hdrs = ["conv_pointwise.h"],
    deps = [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/selectors:subgraph",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "conv_pointwise_test",
    srcs = ["conv_pointwise_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    # TODO(b/279977471) Once b/279347631 is resolved, check for heap again
    deps = [
        ":conv_pointwise",
        "//tflite/delegates/gpu/cl/kernels:cl_test",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:testing_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_library(
    name = "dw7x7_conv2to6_concat_conv8to8",
    srcs = ["dw7x7_conv2to6_concat_conv8to8.cc"],
    hdrs = ["dw7x7_conv2to6_concat_conv8to8.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:flops_util",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common/selectors:subgraph",
        "//tflite/delegates/gpu/common/task:qcom_thin_filter_desc",
    ],
)

cc_library(
    name = "fc_fc_add",
    srcs = ["fc_fc_add.cc"],
    hdrs = ["fc_fc_add.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/selectors:subgraph",
        "//tflite/delegates/gpu/common/task:buffer_desc",
        "//tflite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "thin_pointwise_fuser",
    srcs = ["thin_pointwise_fuser.cc"],
    hdrs = ["thin_pointwise_fuser.h"],
    deps = [
        "//tflite/delegates/gpu/common:flops_util",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/selectors:subgraph",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        "//tflite/delegates/gpu/common/task:util",
        "//tflite/delegates/gpu/common/tasks:elementwise",
        "//tflite/delegates/gpu/common/tasks:prelu",
        "//tflite/delegates/gpu/common/tasks:relu",
        "@com_google_absl//absl/strings",
    ],
)
