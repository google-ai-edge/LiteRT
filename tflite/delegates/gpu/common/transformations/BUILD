load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_platform_alias")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "add_bias",
    srcs = ["add_bias.cc"],
    hdrs = ["add_bias.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "add_quant_adjustments",
    srcs = ["add_quant_adjustments.cc"],
    hdrs = ["add_quant_adjustments.h"],
    deps = [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "add_quant_adjustments_test",
    srcs = ["add_quant_adjustments_test.cc"],
    deps = [
        ":add_quant_adjustments",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fuse_add_to_conv",
    srcs = ["fuse_add_to_conv.cc"],
    hdrs = ["fuse_add_to_conv.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "fuse_add_to_conv_test",
    srcs = ["fuse_add_to_conv_test.cc"],
    deps = [
        ":fuse_add_to_conv",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fuse_mul_to_conv",
    srcs = ["fuse_mul_to_conv.cc"],
    hdrs = ["fuse_mul_to_conv.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "fuse_mul_to_conv_test",
    srcs = ["fuse_mul_to_conv_test.cc"],
    deps = [
        ":fuse_mul_to_conv",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "global_pooling_to_reduce_op",
    srcs = ["global_pooling_to_reduce_op.cc"],
    hdrs = ["global_pooling_to_reduce_op.h"],
    deps = [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "global_pooling_to_reduce_op_test",
    srcs = ["global_pooling_to_reduce_op_test.cc"],
    deps = [
        ":global_pooling_to_reduce_op",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "make_fully_connected",
    srcs = ["make_fully_connected.cc"],
    hdrs = ["make_fully_connected.h"],
    deps = [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "make_fully_connected_test",
    srcs = ["make_fully_connected_test.cc"],
    deps = [
        ":make_fully_connected",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "make_padding",
    srcs = ["make_padding.cc"],
    hdrs = ["make_padding.h"],
    deps = [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "make_padding_test",
    srcs = ["make_padding_test.cc"],
    deps = [
        ":make_padding",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matching",
    hdrs = ["matching.h"],
)

cc_library(
    name = "merge_densify",
    srcs = ["merge_densify.cc"],
    hdrs = ["merge_densify.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "merge_padding_with",
    srcs = ["merge_padding_with.cc"],
    hdrs = ["merge_padding_with.h"],
    deps = [
        ":matching",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "merge_padding_with_test",
    srcs = ["merge_padding_with_test.cc"],
    deps = [
        ":merge_padding_with",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_transformations",
    srcs = ["model_transformations.cc"],
    hdrs = ["model_transformations.h"],
    deps = [
        ":add_quant_adjustments",
        ":fuse_add_to_conv",
        ":fuse_mul_to_conv",
        ":make_fully_connected",
        ":make_padding",
        ":merge_densify",
        ":merge_padding_with",
        ":remove_noop",
        "//tflite/delegates/gpu/common:model_transformer",
    ] + tf_platform_alias("custom_transformations", "//tflite/delegates/gpu/common/"),
)

cc_library(
    name = "remove_noop",
    srcs = ["remove_noop.cc"],
    hdrs = ["remove_noop.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "remove_noop_test",
    srcs = ["remove_noop_test.cc"],
    deps = [
        ":remove_noop",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
