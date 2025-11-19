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
    hdrs = [
        "op_version.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite/converter/tools/versioning",
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
        "//tflite:stderr_reporter",
        "//tflite/converter/tools/versioning:op_signature",
        "//tflite/core/api",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
    ],
)

tf_cc_test(
    name = "op_signature_test",
    srcs = [
        "op_signature_test.cc",
    ],
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/multi_signatures.bin",
    ],
    deps = [
        ":op_signature",
        "//tflite:builtin_ops",
        "//tflite/core:model_builder",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)

cc_library(
    name = "gpu_compatibility",
    srcs = [
        "gpu_compatibility.cc",
    ],
    hdrs = [
        "gpu_compatibility.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//tflite:builtin_op_data",
        "//tflite:builtin_ops",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "gpu_compatibility_test",
    srcs = [
        "gpu_compatibility_test.cc",
    ],
    data = [
        "//tflite:testdata/conv3d_huge_im2col.bin",
        "//tflite:testdata/conv_huge_im2col.bin",
        "//tflite:testdata/multi_add_flex.bin",
    ],
    deps = [
        ":gpu_compatibility",
        ":op_signature",
        "//tflite/core:model_builder",
        "//tflite/core/c:common",
        "//tflite/kernels/internal:types",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)
