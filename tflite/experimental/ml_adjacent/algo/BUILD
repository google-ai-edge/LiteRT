load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "crop",
    srcs = ["crop.cc"],
    hdrs = [
        "crop.h",
    ],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "crop_test",
    srcs = ["crop_test.cc"],
    deps = [
        ":crop",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    deps = [
        ":resize",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rotate",
    srcs = ["rotate.cc"],
    hdrs = ["rotate.h"],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "rotate_test",
    srcs = ["rotate_test.cc"],
    deps = [
        ":rotate",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flip_left_right",
    srcs = ["flip_left_right.cc"],
    hdrs = ["flip_left_right.h"],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "flip_left_right_test",
    srcs = ["flip_left_right_test.cc"],
    deps = [
        ":flip_left_right",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flip_up_down",
    srcs = ["flip_up_down.cc"],
    hdrs = ["flip_up_down.h"],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "flip_up_down_test",
    srcs = ["flip_up_down_test.cc"],
    deps = [
        ":flip_up_down",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "per_image_standardization",
    srcs = ["per_image_standardization.cc"],
    hdrs = ["per_image_standardization.h"],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "per_image_standardization_test",
    srcs = ["per_image_standardization_test.cc"],
    deps = [
        ":per_image_standardization",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rgb_to_grayscale",
    srcs = ["rgb_to_grayscale.cc"],
    hdrs = ["rgb_to_grayscale.h"],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "rgb_to_grayscale_test",
    srcs = ["rgb_to_grayscale_test.cc"],
    deps = [
        ":rgb_to_grayscale",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rgb_to_yuv",
    srcs = ["rgb_to_yuv.cc"],
    hdrs = [
        "image_utils.h",
        "rgb_to_yuv.h",
    ],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "rgb_to_yuv_test",
    srcs = ["rgb_to_yuv_test.cc"],
    deps = [
        ":rgb_to_yuv",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "yuv_to_rgb",
    srcs = ["yuv_to_rgb.cc"],
    hdrs = [
        "image_utils.h",
        "yuv_to_rgb.h",
    ],
    deps = [
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "yuv_to_rgb_test",
    srcs = ["yuv_to_rgb_test.cc"],
    deps = [
        ":yuv_to_rgb",
        "//tflite/experimental/ml_adjacent:lib",
        "//tflite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)
