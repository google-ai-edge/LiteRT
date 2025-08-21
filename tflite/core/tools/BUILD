load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_self_contained_libs_test_suite")
load("//tflite:special_rules.bzl", "verifier_internal_visibility_allowlist")
load("//tflite/core:special_rules.bzl", "verifier_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite:__subpackages__",
    ] + verifier_visibility_allowlist(),
    deps = [
        "//tflite:framework",
        "//tflite:schema_fbs_version",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/converter/schema:schema_utils",
        "//tflite/core:framework",
        "//tflite/core/api:error_reporter",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:common",
        "//tflite/core/tools:verifier_internal",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "verifier_test",
    size = "small",
    srcs = ["verifier_test.cc"],
    data = [
        "//tflite:testdata/sparse_tensor.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":verifier",
        "//tflite:framework",
        "//tflite:schema_fbs_version",
        "//tflite:string",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)

cc_library(
    name = "verifier_internal",
    srcs = ["verifier_internal.cc"],
    hdrs = ["verifier_internal.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = verifier_internal_visibility_allowlist() + ["//tflite/tools:__subpackages__"],
    deps = [
        "//tflite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "verifier_internal_test",
    size = "small",
    srcs = ["verifier_internal_test.cc"],
    deps = [
        ":verifier_internal",
        "//tflite:framework",
        "//tflite:schema_fbs_version",
        "//tflite/c:common",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
