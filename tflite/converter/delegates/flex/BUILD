load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "if_mobile",
    "if_not_mobile",
    "tf_features_nolayering_check_if_ios",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:special_rules.bzl", "internal_visibility_allowlist")

default_visibility = [
    "//tflite/converter:__subpackages__",
    "//tflite/android:__subpackages__",
    "//tflite/toco/tflite:__subpackages__",
]

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

cc_library(
    name = "allowlisted_flex_ops_lib",
    srcs = [
        "allowlisted_flex_ops.cc",
    ],
    hdrs = [
        "allowlisted_flex_ops.h",
        "allowlisted_flex_ops_internal.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = select({
        "@org_tensorflow//tensorflow:android": ["-Wno-private-header"],
        "//conditions:default": [],
    }),
    features = tf_features_nolayering_check_if_ios(),
    visibility = internal_visibility_allowlist(),
    deps = if_mobile([
        "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "@org_tensorflow//tensorflow/core:framework",
    ]),
)
