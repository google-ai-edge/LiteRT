load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Test utilities for TFLite async execution.
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "test_backend",
    testonly = 1,
    srcs = ["test_backend.cc"],
    hdrs = ["test_backend.h"],
    deps = [
        "//tflite:array",
        "//tflite:builtin_ops",
        "//tflite/core/async:async_kernel_internal",
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/delegates:utils",
    ],
)

cc_library_with_tflite(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    deprecation = "Use //tensorflow/lite/async/testing:mock_async_kernel instead.",
    tflite_deps = [
        "//tflite/async/testing:mock_async_kernel",
    ],
)
