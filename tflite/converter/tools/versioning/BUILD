load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "versioning",
    srcs = [
        "op_version.cc",
        "runtime_version.cc",
    ],
    hdrs = [
        "op_version.h",
        "runtime_version.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//tflite/converter/core/c:tflite_common",
        "//tflite/converter/kernels/internal:compatibility_macros",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/converter/schema:schema_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "op_version_test",
    srcs = [
        "op_version_test.cc",
    ],
    deps = [
        ":op_signature",
        ":versioning",
        "//tflite/converter/core/c:tflite_common",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "runtime_version_test",
    srcs = [
        "runtime_version_test.cc",
    ],
    deps = [
        ":versioning",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "op_signature",
    srcs = [
        "op_signature.cc",
    ],
    hdrs = [
        "op_signature.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite/converter/core/api:flatbuffer_conversions",
        "//tflite/converter/core/c:tflite_common",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

tf_cc_test(
    name = "op_signature_test",
    srcs = [
        "op_signature_test.cc",
    ],
    data = [
        "//tflite/converter:testdata/add.bin",
        "//tflite/converter:testdata/multi_signatures.bin",
    ],
    deps = [
        ":op_signature",
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/core/c:tflite_common",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)
