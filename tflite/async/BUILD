load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Description:
#
# This package contains shim library targets for the Async C package.
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "backend_async_kernel_interface",
    srcs = ["backend_async_kernel_interface.cc"],
    hdrs = ["backend_async_kernel_interface.h"],
    tflite_deps = [
        "//tflite/async/c:async_kernel",
        "//tflite/async/c:types",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "backend_async_kernel_interface_test",
    srcs = ["backend_async_kernel_interface_test.cc"],
    deps = [
        ":backend_async_kernel_interface",
        "//tflite/async/c:types",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/async:async_kernel_internal",
        "//tflite/core/async/testing:mock_async_kernel",
        "@com_google_googletest//:gtest_main",
    ],
)
