load(
    "@org_tensorflow//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "feature_parity",
    hdrs = ["feature_parity.h"],
    deps = [
        ":utils",
        "//tflite/delegates/gpu/common/testing/feature_parity/generators:all_generators",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//tflite:framework",
        "//tflite:string",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "opengl_test",
    srcs = ["opengl_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":feature_parity",
        ":utils",
        "//tflite:framework_lib",
        "//tflite/delegates/gpu:gl_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "opencl_test",
    srcs = ["opencl_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":feature_parity",
        ":utils",
        "//tflite:framework_lib",
        "//tflite/delegates/gpu:delegate",
        "//tflite/delegates/gpu:delegate_options",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "xnnpack_test",
    srcs = ["xnnpack_test.cc"],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":feature_parity",
        ":utils",
        "//tflite:framework_lib",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)
