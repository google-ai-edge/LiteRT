# Tools to strip out eligible buffers (tensor data) from a TFLite flatbuffer, and load them with
# random values at runtime.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "stripping_lib",
    srcs = ["stripping_lib.cc"],
    hdrs = ["stripping_lib.h"],
    copts = tflite_copts(),
    deps = [
        "//tflite:framework",
        "//tflite:string",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
    ],
)

cc_binary(
    name = "strip_buffers_from_fb",
    srcs = ["strip_buffers_from_fb.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        ":stripping_lib",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/tools:command_line_flags",
        "@com_google_absl//absl/log",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
    ],
)

cc_binary(
    name = "reconstitute_buffers_into_fb",
    srcs = ["reconstitute_buffers_into_fb.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        ":stripping_lib",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/tools:command_line_flags",
        "@com_google_absl//absl/log",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
    ],
)
