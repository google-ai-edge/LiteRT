load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_test(
    name = "gpu_delegate_compatibility_checker_test",
    srcs = [
        "gpu_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/conv3d_huge_im2col.bin",
    ],
    tags = ["no_oss"],  #TODO(b/276295784): Re-enable when fixed.
    deps = [
        ":gpu_delegate_compatibility_checker",
        "//tflite:model_builder",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "//tflite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/platform:resource_loader",
    ],
)

cc_library(
    name = "gpu_delegate_compatibility_checker",
    srcs = ["gpu_delegate_compatibility_checker.cc"],
    hdrs = [
        "gpu_delegate_compatibility_checker.h",
    ],
    deps = [
        "//tflite:model_builder",
        "//tflite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//tflite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tflite/tools/versioning:gpu_compatibility",
        "//tflite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)
