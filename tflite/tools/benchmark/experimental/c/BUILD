load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["benchmark"],
    licenses = ["notice"],
)

package_group(
    name = "benchmark",
    packages = [
        "//tflite/tools/benchmark/...",
    ],
)

exports_files(
    ["benchmark_c_api.h"],
    visibility = ["//tflite/tools/benchmark/experimental/c:benchmark"],
)

cc_library(
    name = "benchmark_c_api",
    srcs = ["benchmark_c_api.cc"],
    hdrs = [
        "benchmark_c_api.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "benchmark",
    ],
    deps = [
        "//tflite/c:c_api_types",
        "//tflite/core/c:c_api_types",
        "//tflite/tools/benchmark:benchmark_model_lib",
        "//tflite/tools/benchmark:benchmark_params",
        "//tflite/tools/benchmark:benchmark_tflite_model_lib",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)
