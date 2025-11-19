# A simple op. for testing and demonstrating the OpKernel interface.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_kernel_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/kernels/shim:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "simple_op",
    hdrs = ["simple_op.h"],
    deps = [
        "//tflite/kernels/shim:op_kernel",
        "//tflite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "simple_tf_op",
    srcs = ["simple_tf_op.cc"],
    hdrs = ["simple_tf_op.h"],
    deps = [
        ":simple_op",
        "//tflite/kernels/shim:tf_op_shim",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "simple_tf_op_test",
    srcs = ["simple_tf_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":simple_tf_op",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "simple_tflite_op",
    srcs = ["simple_tflite_op.cc"],
    hdrs = ["simple_tflite_op.h"],
    deps = [
        ":simple_op",
        "//tflite:mutable_op_resolver",
        "//tflite/core/c:common",
        "//tflite/kernels/shim:tflite_op_shim",
    ],
)

cc_test(
    name = "simple_tflite_op_test",
    srcs = ["simple_tflite_op_test.cc"],
    deps = [
        ":simple_tflite_op",
        "//tflite/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "tmpl_op",
    hdrs = ["tmpl_op.h"],
    deps = [
        "//tflite/kernels/shim:op_kernel",
        "//tflite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "tmpl_tf_op",
    srcs = ["tmpl_tf_op.cc"],
    hdrs = ["tmpl_tf_op.h"],
    deps = [
        ":tmpl_op",
        "//tflite/kernels/shim:tf_op_shim",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "tmpl_tflite_op",
    srcs = ["tmpl_tflite_op.cc"],
    hdrs = ["tmpl_tflite_op.h"],
    deps = [
        ":tmpl_op",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/kernels/shim:op_kernel",
        "//tflite/kernels/shim:tflite_op_shim",
        "//tflite/kernels/shim:tflite_op_wrapper",
    ],
)

tf_cc_test(
    name = "tmpl_tf_op_test",
    srcs = ["tmpl_tf_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":tmpl_tf_op",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/framework:fake_input",
        "@org_tensorflow//tensorflow/core/framework:tensor_testutil",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_cc_test(
    name = "tmpl_tflite_op_test",
    srcs = ["tmpl_tflite_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":tmpl_tflite_op",
        "//tflite/c:c_api_types",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

tflite_portable_test_suite()
