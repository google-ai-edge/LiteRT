load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "online_helper_delegate",
    srcs = ["online_helper_delegate.cc"],
    hdrs = ["online_helper_delegate.h"],
    deps = [
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_base",
    srcs = ["delegate_compatibility_checker_base.cc"],
    hdrs = [
        "delegate_compatibility_checker_base.h",
    ],
    deps = [
        ":delegate_compatibility_checker_util",
        "//tflite:framework_stable",
        "//tflite/core:framework_stable",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "//tflite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tflite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_util",
    hdrs = [
        "delegate_compatibility_checker_util.h",
    ],
)
