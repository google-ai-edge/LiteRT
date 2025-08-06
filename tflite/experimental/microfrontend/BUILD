# TensorFlow ops for audio front-end processing.

load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_opts_nortti_if_android",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "tf_custom_op_py_strict_library", "tf_py_strict_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "audio_microfrontend",
    srcs = ["audio_microfrontend.cc"],
    hdrs = ["audio_microfrontend.h"],
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/experimental/microfrontend/lib:frontend",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/kernels/internal:reference",
        "@flatbuffers",
    ],
)

cc_library(
    name = "audio_microfrontend_op_lib",
    srcs = ["ops/audio_microfrontend_op.cc"],
    copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_android() + [
        "-Wno-narrowing",
        "-Wno-sign-compare",
        "-Wno-overloaded-virtual",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            # Selective registration uses constexprs with recursive
            # string comparisons; that can lead to compiler errors, so
            # we increase the constexpr recursion depth.
            "-fconstexpr-depth=1024",
            "-Oz",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tflite/experimental/microfrontend/lib:frontend",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "audio_microfrontend_test",
    size = "small",
    srcs = ["audio_microfrontend_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":audio_microfrontend",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tf_custom_op_library(
    name = "python/ops/_audio_microfrontend_op.so",
    srcs = [
        "ops/audio_microfrontend_op.cc",
    ],
    deps = [
        "//tflite/experimental/microfrontend/lib:frontend",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["audio_microfrontend_op"],
    deps = [
        "//tflite/experimental/microfrontend/lib:frontend",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "audio_microfrontend_op",
    extra_py_deps = [
        "@org_tensorflow//tensorflow/python:pywrap_tfe",
        "@org_tensorflow//tensorflow/python/util:dispatch",
        "@org_tensorflow//tensorflow/python/util:deprecation",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":audio_microfrontend_op_op_lib"],
)

tf_custom_op_py_strict_library(
    name = "audio_microfrontend_py",
    srcs = [
        "python/ops/audio_microfrontend_op.py",
    ],
    dso = [":python/ops/_audio_microfrontend_op.so"],
    kernels = [
        ":audio_microfrontend_op_op_lib",
    ],
    deps = [
        ":audio_microfrontend_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:load_library",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:control_flow_ops",
        "@org_tensorflow//tensorflow/python/ops:linalg_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

tf_py_strict_test(
    name = "audio_microfrontend_op_test",
    size = "small",
    srcs = ["python/kernel_tests/audio_microfrontend_op_test.py"],
    deps = [
        ":audio_microfrontend_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/framework:ops",
    ],
)
