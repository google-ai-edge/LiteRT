load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite:special_rules.bzl", "op_resolver_internal_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        ":error_reporter.h",
        ":op_resolver.h",
        ":op_resolver_internal.h",
        ":verifier.h",
    ],
    visibility = [
        "//tflite:__pkg__",
    ],
)

cc_library(
    name = "api",
    srcs = [
        "flatbuffer_conversions.cc",
        "tensor_utils.cc",
    ],
    hdrs = [
        "error_reporter.h",
        "flatbuffer_conversions.h",
        "op_resolver.h",
        "profiler.h",
        "tensor_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":error_reporter",
        ":op_resolver",
        "@flatbuffers//:runtime_cc",
        "//tflite/core/c:common",
        # TODO(b/158301698): consider moving internal:compatibility to a more
        # central location.
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
    ],
)

# We define separate targets for "op_resolver" and "error_reporter",
# even though those headers are also exported by the "api" target,
# so that targets which only want to depend on these small abstract base
# class modules can express more fine-grained dependencies without
# pulling in tensor_utils and flatbuffer_conversions.

cc_library(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":error_reporter",
        "//tflite/converter/schema:schema_utils",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "error_reporter",
    hdrs = [
        "error_reporter.h",
        "//tflite/converter/core/api:error_reporter.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tflite/converter/core/api:error_reporter",
    ],
)

cc_library(
    name = "verifier",
    hdrs = [
        "verifier.h",
        "//tflite/converter/core/api:verifier.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/converter/core/api:error_reporter",
        "//tflite/converter/core/api:verifier",
    ],
)

cc_library(
    name = "op_resolver_internal",
    hdrs = ["op_resolver_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = op_resolver_internal_visibility_allowlist() + [
        "//tflite:__pkg__",
        "//tflite/java/src/main/native:__pkg__",
    ],
    deps = [":op_resolver"],
)

cc_test(
    name = "op_resolver_test",
    size = "small",
    srcs = ["op_resolver_test.cc"],
    deps = [
        ":api",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/converter/core/api:error_reporter",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "op_resolver_internal_test",
    size = "small",
    srcs = ["op_resolver_internal_test.cc"],
    deps = [
        ":op_resolver",
        ":op_resolver_internal",
        "//tflite:framework",
        "//tflite:mutable_op_resolver",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "flatbuffer_conversions_test",
    size = "small",
    srcs = ["flatbuffer_conversions_test.cc"],
    deps = [
        ":api",
        "//tflite:string",
        "//tflite/c:c_api_types",
        "//tflite/converter/core/api:error_reporter",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)
