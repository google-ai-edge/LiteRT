load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_status",
        "//tflite/core/api",
        "//tflite/profiling/telemetry/c:profiler",
        "//tflite/profiling/telemetry/c:telemetry_setting",
    ],
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry.cc"],
    hdrs = ["telemetry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_status",
        "//tflite/core/api",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/profiling/telemetry/c:telemetry_setting",
    ],
)

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    deps = [
        ":profiler",
        ":telemetry",
        ":telemetry_status",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/profiling/telemetry/c:telemetry_setting_internal",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    deps = [
        ":profiler",
        ":telemetry",
        ":telemetry_status",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/profiling/telemetry/c:telemetry_setting",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "telemetry_status",
    hdrs = ["telemetry_status.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = ["//tflite/core/c:c_api_types"],
)
