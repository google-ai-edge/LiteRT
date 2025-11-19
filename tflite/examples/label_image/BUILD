# Description:
# TensorFlow Lite Example Label Image.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bmp",
]))

cc_binary(
    name = "label_image",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,-rpath=/data/local/tmp",  # for hexagon delegate
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":bitmap_helpers",
        "//tflite:framework",
        "//tflite:string",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/kernels:builtin_ops",
        "//tflite/profiling:profile_buffer",
        "//tflite/profiling:profiler",
        "//tflite/schema:schema_fbs",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:tool_params",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "//tflite/tools/delegates:tflite_execution_providers",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
        "log.h",
    ],
    deps = [
        "//tflite:builtin_op_data",
        "//tflite:framework",
        "//tflite:string",
        "//tflite:string_util",
        "//tflite/c:c_api_types",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@local_tsl//tsl/platform:tstring",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu:delegate",
            "//tflite/delegates/hexagon:hexagon_delegate",
        ],
        "@org_tensorflow//tensorflow:android_arm64": [
            "//tflite/delegates/gpu:delegate",
            "//tflite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "label_image_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)
