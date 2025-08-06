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
    name = "simple_delegate",
    srcs = [
        "simple_delegate.cc",
    ],
    hdrs = [
        "simple_delegate.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:array",
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite/core/c:common",
        "//tflite/delegates:utils",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "simple_delegate_test",
    srcs = ["simple_delegate_test.cc"],
    deps = [
        "//tflite:framework",
        "//tflite:kernel_api",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/utils/dummy_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "simple_opaque_delegate",
    srcs = ["simple_opaque_delegate.cc"],
    hdrs = ["simple_opaque_delegate.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//tflite/c:c_api",
        "//tflite/c:c_api_experimental",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
    ],
    deps = [
        "//tflite:array",
        "//tflite:builtin_ops",
        "//tflite:minimal_logging",
        "//tflite:util",
        "//tflite/kernels/internal:compatibility",
    ],
)

filegroup(
    name = "c_api_test_builtin_op_models",
    testonly = 1,
    srcs = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/conv_huge_im2col.bin",
        "//tflite:testdata/multi_add.bin",
    ],
)

cc_test(
    name = "simple_opaque_delegate_test",
    srcs = ["simple_opaque_delegate_test.cc"],
    data = [":c_api_test_builtin_op_models"],
    deps = [
        ":simple_opaque_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:c_api",
        "//tflite/c:c_api_experimental",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates:delegate_test_util",
        "//tflite/delegates/utils/experimental/sample_stable_delegate",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:compatibility",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "async_type_helpers",
    srcs = ["async_type_helpers.cc"],
    hdrs = ["async_type_helpers.h"],
    tflite_deps = [
        ":ret_macros",
        "//tflite/async/interop/c:attribute_map",
        "//tflite/async/interop/c:constants",
        "//tflite/async/interop/c:types",
    ],
)

cc_library_with_tflite(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    tflite_deps = [
        "//tflite/c:c_api_experimental",
        "//tflite/c:c_api_types",
        ":ret_macros",
    ],
    deps = [
        "//tflite:array",
        "//tflite:minimal_logging",
        "@com_google_absl//absl/status",
    ],
)

cc_library_with_tflite(
    name = "ret_macros",
    srcs = [],
    hdrs = ["ret_macros.h"],
    tflite_deps = ["//tflite/c:c_api_types"],
    deps = [
        "//tflite:minimal_logging",
    ],
)

cc_library_with_tflite(
    name = "sync_fence",
    srcs = ["sync_fence.cc"],
    hdrs = ["sync_fence.h"],
    tflite_deps = [
        ":ret_macros",
    ],
    deps = [
        "//tflite:minimal_logging",
        "@com_google_absl//absl/types:span",
    ],
)
