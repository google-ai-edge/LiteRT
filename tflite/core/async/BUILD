load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "async_kernel_internal",
    hdrs = ["async_kernel_internal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_library(
    name = "task_internal",
    srcs = ["task_internal.cc"],
    hdrs = ["task_internal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        "//tflite/core/async/c:types",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "task_internal_test",
    srcs = ["task_internal_test.cc"],
    deps = [
        ":task_internal",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/async/c:types",
        "//tflite/core/async/interop/c:types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "async_subgraph",
    srcs = ["async_subgraph.cc"],
    hdrs = ["async_subgraph.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":task_internal",
        "//tflite:minimal_logging",
        "//tflite/core:subgraph",
        "//tflite/core/async/c:types",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "async_subgraph_test",
    srcs = ["async_subgraph_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":async_subgraph",
        ":backend_async_kernel_interface",
        ":task_internal",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/async/c:types",
        "//tflite/core/async/interop:attribute_map_internal",
        "//tflite/core/async/interop/c:types",
        "//tflite/core/async/testing:mock_async_kernel",
        "//tflite/core/async/testing:test_backend",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "backend_async_kernel_interface",
    hdrs = ["backend_async_kernel_interface.h"],
    deprecation = "Use //tensorflow/lite/async:backend_async_kernel_interface instead.",
    tflite_deps = [
        "//tflite/async:backend_async_kernel_interface",
    ],
)

cc_library(
    name = "async_signature_runner",
    srcs = ["async_signature_runner.cc"],
    hdrs = ["async_signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":async_subgraph",
        ":task_internal",
        "//tflite/core:subgraph",
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/internal:signature_def",
    ],
)

cc_test(
    name = "async_signature_runner_test",
    srcs = ["async_signature_runner_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":async_signature_runner",
        ":backend_async_kernel_interface",
        "//tflite:framework",
        "//tflite:interpreter_test_util",
        "//tflite/core:framework_stable",
        "//tflite/core/async/c:task",
        "//tflite/core/async/c:types",
        "//tflite/core/async/interop:attribute_map_internal",
        "//tflite/core/async/testing:mock_async_kernel",
        "//tflite/core/async/testing:test_backend",
        "//tflite/core/c:c_api_experimental",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)
