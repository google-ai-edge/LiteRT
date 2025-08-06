load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# BUILD rules for NNAPI delegate compatibility checking.
package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_compatibility_lib",
    srcs = [
        "nnapi_compatibility_lib.cc",
    ],
    hdrs = [
        "nnapi_compatibility_lib.h",
    ],
    deps = [
        "//tflite:framework_stable",
        "//tflite:minimal_logging",
        "//tflite/core/c:common",
        "//tflite/delegates/nnapi:nnapi_delegate",
    ],
)

cc_library(
    name = "nnapi_delegate_compatibility_checker",
    srcs = ["nnapi_delegate_compatibility_checker.cc"],
    hdrs = [
        "nnapi_delegate_compatibility_checker.h",
    ],
    copts = ["-DNNAPI_VERBOSE_VALIDATION"],
    deps = [
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/nnapi:nnapi_delegate",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//tflite/tools/delegates/compatibility/common:delegate_compatibility_checker_util",
        "//tflite/tools/delegates/compatibility/common:online_helper_delegate",
        "//tflite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tflite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "nnapi_compatibility_lib_test",
    srcs = [
        "nnapi_compatibility_lib_test.cc",
    ],
    deps = [
        ":nnapi_compatibility_lib",
        "//tflite/core/c:c_api_types",
        "//tflite/delegates/nnapi:nnapi_delegate_verbose_validation",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_compatibility_checker_test",
    srcs = [
        "nnapi_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//tflite:testdata/add.bin",
    ],
    tags = ["no_oss"],  #TODO(b/276295784): Re-enable when fixed.
    deps = [
        ":nnapi_delegate_compatibility_checker",
        "//tflite:model_builder",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "//tflite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)
