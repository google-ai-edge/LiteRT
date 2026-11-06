load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tfl_dynamic_histogram",
    srcs = ["tfl_dynamic_histogram.cc"],
    hdrs = ["tfl_dynamic_histogram.h"],
)

cc_test(
    name = "tfl_dynamic_histogram_test",
    srcs = ["tfl_dynamic_histogram_test.cc"],
    deps = [
        ":tfl_dynamic_histogram",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tfl_calibration_utils",
    srcs = ["tfl_calibration_utils.cc"],
    hdrs = ["tfl_calibration_utils.h"],
    deps = [
        ":tfl_dynamic_histogram",
        ":tfl_tensor_stats_profiler",
        "//tflite:framework",
        "//tflite:stateful_error_reporter",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/profiling:memory_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "tfl_tensor_stats_profiler",
    srcs = ["tfl_tensor_stats_profiler.cc"],
    hdrs = ["tfl_tensor_stats_profiler.h"],
    deps = [
        "//tflite:framework",
        "//tflite:minimal_logging",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/c:private_common",
    ],
)

cc_test(
    name = "tfl_tensor_stats_profiler_test",
    srcs = ["tfl_tensor_stats_profiler_test.cc"],
    deps = [
        ":tfl_tensor_stats_profiler",
        "//tflite:framework",
        "//tflite/core/api",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tfl_calibration_utils_test",
    srcs = ["tfl_calibration_utils_test.cc"],
    tags = ["nohwasan"],
    deps = [
        ":tfl_calibration_utils",
        "//tflite:framework",
        "//tflite/profiling:memory_info",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

pybind_extension(
    name = "_pywrap_tfl_calibration",
    srcs = ["tfl_calibration_wrapper.cc"],
    module_name = "_pywrap_tfl_calibration",
    wrap_py_init = True,
    deps = [
        ":tfl_calibration_utils",
        "//tflite:framework",
        "//tflite:stateful_error_reporter",
        "//tflite/profiling:memory_info",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)
