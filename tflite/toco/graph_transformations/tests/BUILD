load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "lstm_utils_test",
    srcs = ["lstm_utils_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "identify_l2_pool_test",
    srcs = ["identify_l2_pool_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "remove_successive_transpose_test",
    srcs = ["remove_successive_transpose_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_concatenation_test",
    srcs = ["resolve_constant_concatenation_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "unpack_quantize_test",
    srcs = ["unpack_quantize_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_unary_test",
    srcs = ["resolve_constant_unary_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "identify_l2_normalization_test",
    srcs = ["identify_l2_normalization_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "fuse_binary_into_following_affine_test",
    srcs = ["fuse_binary_into_following_affine_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "fuse_binary_into_preceding_affine_test",
    srcs = ["fuse_binary_into_preceding_affine_test.cc"],
    deps = [
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)
