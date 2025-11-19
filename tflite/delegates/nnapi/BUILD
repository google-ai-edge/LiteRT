load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_cc_library_with_c_headers_test")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Client of this library is responsible to provide an implementation of
# for headers in //third_party/tensorflow/lite/nnapi:nnapi_implementation_headers
cc_library(
    name = "nnapi_delegate_no_nnapi_implementation",
    srcs = select({
        "@org_tensorflow//tensorflow:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
            "nnapi_delegate_c_api.cc",
            "quant_lstm_sup.cc",
            "quant_lstm_sup.h",
        ],
    }),
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_c_api.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/delegates:serialization",
        "//tflite/delegates:utils",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal/utils:sparsity_format_converter",
        "//tflite/nnapi:nnapi_implementation_headers",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/nnapi:nnapi_util",
        "//tflite/nnapi/sl:nnapi_support_library_headers",
        "//tflite/schema:schema_fbs",
        "@FP16",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "nnapi_delegate",
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_c_api.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nnapi_delegate_no_nnapi_implementation",
        "//tflite:allocation",
        "//tflite/core/c:common",
        "//tflite/delegates:serialization",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "nnapi_delegate_c_api",
    hdrs = ["nnapi_delegate_c_api.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        ":nnapi_delegate",
        "//tflite/core/c:common",
    ],
)

exports_files(["nnapi_delegate_c_api.h"])

cc_library(
    name = "nnapi_delegate_verbose_validation",
    srcs = select({
        "@org_tensorflow//tensorflow:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
            "quant_lstm_sup.cc",
            "quant_lstm_sup.h",
        ],
    }),
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    copts = ["-DNNAPI_VERBOSE_VALIDATION"],
    deps = [
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/delegates:serialization",
        "//tflite/delegates:utils",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal/utils:sparsity_format_converter",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/nnapi:nnapi_util",
        "//tflite/nnapi/sl:nnapi_support_library_headers",
        "//tflite/schema:schema_fbs",
        "@FP16",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "acceleration_test_util",
    testonly = 1,
    srcs = [
        "acceleration_test_list.cc",
        "acceleration_test_util.cc",
    ],
    hdrs = ["acceleration_test_util.h"],
    deps = [
        ":nnapi_delegate",
        "//tflite/kernels:acceleration_test_util_internal",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "nnapi_delegate_mock_test",
    testonly = 1,
    hdrs = [
        "nnapi_delegate_mock_test.h",
    ],
    copts = select({
        "@org_tensorflow//tensorflow:ios": [
            "-DNNAPI_DELEGATE_DISABLED",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "-DNNAPI_DELEGATE_DISABLED",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":nnapi_delegate",
        "//tflite/nnapi:nnapi_handler",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "nnapi_delegate_test",
    size = "small",
    srcs = [
        "nnapi_delegate_test.cc",
    ],
    tags = [
        "no_windows",
        # TODO(b/401046768): Re-enable once the test is fixed.
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        "//tflite:framework",
        "//tflite:string",
        "//tflite:string_util",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:deprecated_backends",
        "//tflite/kernels:test_util",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_c_api_test",
    size = "small",
    srcs = [
        "nnapi_delegate_c_api_test.cc",
    ],
    tags = [
        "no_windows",
        # TODO(b/401046768): Re-enable once the test is fixed.
        "notap",
        "tflite_not_portable_ios",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":nnapi_delegate",
        "//tflite:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:deprecated_backends",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_errno_test",
    size = "small",
    srcs = [
        "nnapi_delegate_errno_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_device_selection_test",
    size = "small",
    srcs = [
        "nnapi_delegate_device_selection_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tflite:framework",
        "//tflite:kernel_api",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_nnapi_failure_handling_test",
    size = "small",
    srcs = [
        "nnapi_delegate_nnapi_failure_handling_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_signed_quantization_test",
    size = "small",
    srcs = [
        "nnapi_delegate_signed_quantization_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:test_util",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quant_lstm_sup_test",
    size = "small",
    srcs = [
        "quant_lstm_sup.cc",
        "quant_lstm_sup.h",
        "quant_lstm_sup_test.cc",
    ],
    deps = [
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
