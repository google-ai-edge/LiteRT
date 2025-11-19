# Description:
#  Holds the native layer of the app.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "jni_binary_with_tflite")
load("//tflite/tools/benchmark/experimental/delegate_performance/android:build_defs.bzl", "accuracy_benchmark_extra_deps", "latency_benchmark_extra_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/tools/benchmark/experimental/delegate_performance/android:__subpackages__"],
    licenses = ["notice"],
)

jni_binary_with_tflite(
    name = "libdelegate_performance_benchmark.so",
    srcs = ["delegate_performance_benchmark_jni.cc"],
    tflite_deps = [":accuracy_benchmark"],
    deps = [
        ":latency_benchmark",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//tflite/java/jni",
        "//tflite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_cc_proto",
        "@flatbuffers",
    ],
)

cc_library(
    name = "status_codes",
    hdrs = ["status_codes.h"],
)

cc_library(
    name = "latency_benchmark",
    srcs = ["latency_benchmark.cc"],
    hdrs = ["latency_benchmark.h"],
    deps = [
        "//tflite:minimal_logging",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/c:c_api_types",
        "//tflite/profiling:memory_info",
        "//tflite/tools/benchmark:benchmark_model_lib",
        "//tflite/tools/benchmark:benchmark_params",
        "//tflite/tools/benchmark:benchmark_tflite_model_lib",
        "//tflite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core/util:stats_calculator_portable",
    ] + latency_benchmark_extra_deps(),
)

cc_library_with_tflite(
    name = "accuracy_benchmark",
    srcs = ["accuracy_benchmark.cc"],
    hdrs = ["accuracy_benchmark.h"],
    deps = [
        ":status_codes",
        "//tflite:minimal_logging",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/acceleration/configuration:gpu_plugin",
        "//tflite/acceleration/configuration:stable_delegate_plugin",
        "//tflite/acceleration/configuration:xnnpack_plugin",
        "//tflite/core/acceleration/configuration:nnapi_plugin",
        "//tflite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//tflite/experimental/acceleration/mini_benchmark:libjpeg_handle_static_link",
        "//tflite/experimental/acceleration/mini_benchmark:status_codes",
        "//tflite/experimental/acceleration/mini_benchmark/c:c_api_in_process",
        "//tflite/kernels/internal:compatibility",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:tool_params",
        "@flatbuffers",
    ] + accuracy_benchmark_extra_deps(),
)

cc_library_with_tflite(
    name = "benchmark_native",
    tflite_jni_binaries = [":libdelegate_performance_benchmark.so"],
)
