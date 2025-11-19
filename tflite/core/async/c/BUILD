# This package contains the C API libraries for asynchronous execution and buffer interop.
# For clients using async APIs, please use tensorflow/lite/async/c instead of this package.
# NOTE: Targets in this package are experimental.

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tflite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_copts",
    "tflite_copts_warnings",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "types.h",
    ],
    visibility = [
        "//tflite:__subpackages__",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # tensorflow/lite/async/c.
        "//visibility:public",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "async_kernel",
    srcs = ["async_kernel.cc"],
    hdrs = ["async_kernel.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # tensorflow/lite/async/c.
        "//visibility:public",
    ],
    deps = [
        ":types",
        "//tflite/core/async:async_kernel_internal",
        "//tflite/core/async/interop/c:attribute_map",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "task",
    srcs = ["task.cc"],
    hdrs = ["task.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # tensorflow/lite/async/c.
        "//visibility:public",
    ],
    deps = [
        ":types",
        "//tflite/core/async:task_internal",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "task_test",
    srcs = ["task_test.cc"],
    deps = [
        ":task",
        ":types",
        "//tflite/core/async:task_internal",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "async_signature_runner",
    srcs = ["async_signature_runner.cc"],
    hdrs = ["async_signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # tensorflow/lite/async/c.
        "//visibility:public",
    ],
    deps = [
        ":internal",
        ":types",
        "//tflite/c:c_api_internal",
        "//tflite/core/async:async_signature_runner",
        "//tflite/core/async/interop/c:attribute_map",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/c:c_api",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "async_signature_runner_test",
    srcs = ["async_signature_runner_test.cc"],
    copts = tflite_copts() + tflite_copts_warnings(),
    data = [
        "//tflite:testdata/no_signatures.bin",
    ],
    deps = [
        ":async_signature_runner",
        ":internal",
        ":task",
        ":types",
        "//tflite:interpreter_test_util",
        "//tflite/c:c_api_experimental",
        "//tflite/c:c_api_internal",
        "//tflite/core:cc_api_stable",
        "//tflite/core/async:async_kernel_internal",
        "//tflite/core/async:backend_async_kernel_interface",
        "//tflite/core/async/interop/c:attribute_map",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/async/testing:mock_async_kernel",
        "//tflite/core/async/testing:test_backend",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:c_api_without_op_resolver",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "internal",
    hdrs = ["internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:private"],
    deps = ["//tflite/core/async:async_signature_runner"],
)
