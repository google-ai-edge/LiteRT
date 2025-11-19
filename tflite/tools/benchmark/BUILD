load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

# We create a library for benchmark_main.cc to faciliate the creation of a
# customized benchmark model binary that only needs linking with extra
# dependency, e.g., enabling creating of benchmark binaries with a custom
# delegate provider.
cc_library(
    name = "benchmark_model_main",
    srcs = [
        "benchmark_main.cc",
    ],
    copts = common_copts,
    deps = [
        ":benchmark_tflite_model_lib",
        "//tflite/tools:logging",
    ],
)

cc_binary(
    name = "benchmark_model",
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":benchmark_model_main",
    ],
)

cc_binary(
    name = "benchmark_model_performance_options",
    srcs = [
        "benchmark_tflite_performance_options_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":benchmark_performance_options",
        ":benchmark_tflite_model_lib",
        "//tflite/tools:logging",
    ],
)

# As with most target binaries that use flex, this should be built with the
# `--config=monolithic` build flag, e.g.,
#    bazel build --config=monolithic --config=android_arm64 \
#        -c opt --cxxopt='--std=c++17' :benchmark_model_plus_flex
tf_cc_binary(
    name = "benchmark_model_plus_flex",
    srcs = [
        "benchmark_plus_flex_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_tflite_model_lib",
        "//tflite/delegates/flex:delegate",
        "//tflite/testing:init_tensorflow",
        "//tflite/tools:logging",
    ],
)

cc_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.cc"],
    args = [
        "--fp32_graph=$(location //tflite:testdata/multi_add.bin)",
        "--int8_graph=$(location //tflite:testdata/add_quantized_int8.bin)",
        "--string_graph_with_signature=$(location //tflite:testdata/string_input_model_with_signature.bin)",
        "--string_graph_without_signature=$(location //tflite:testdata/string_input_model.bin)",
        "--multi_signature_graph=$(location //tflite:testdata/multi_signatures.bin)",
    ],
    data = [
        "//tflite:testdata/add_quantized_int8.bin",
        "//tflite:testdata/multi_add.bin",
        "//tflite:testdata/multi_signatures.bin",
        "//tflite:testdata/string_input_model.bin",
        "//tflite:testdata/string_input_model_with_signature.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":benchmark_performance_options",
        ":benchmark_tflite_model_lib",
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/testing:util",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools/benchmark/proto:benchmark_result_cc",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "profiling_listener",
    srcs = ["profiling_listener.cc"],
    hdrs = ["profiling_listener.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        "//tflite:framework_stable",
        "//tflite/profiling:profile_summarizer",
        "//tflite/profiling:profile_summary_formatter",
        "//tflite/profiling:profiler",
        "//tflite/tools:logging",
    ],
)

cc_library(
    name = "benchmark_tflite_model_lib",
    srcs = ["benchmark_tflite_model.cc"],
    hdrs = ["benchmark_tflite_model.h"],
    copts = common_copts + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":benchmark_utils",
        ":profiling_listener",
        "//tflite:framework",
        "//tflite:simple_memory_arena_debug_dump",
        "//tflite:string_util",
        "//tflite/core:cc_api_stable",
        "//tflite/core:framework",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels:cpu_backend_context",
        "//tflite/profiling:model_runtime_info",
        "//tflite/profiling:profile_summary_formatter",
        "//tflite/profiling:profiler",
        "//tflite/tools:logging",
        "//tflite/tools:model_loader",
        "//tflite/tools:utils",
        "//tflite/tools/benchmark/proto:benchmark_result_cc",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "//tflite/tools/delegates:tflite_execution_providers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core/example:example_protos_cc_impl",
        "@ruy//ruy/profiler",
    ],
)

cc_test(
    name = "benchmark_tflite_model_lib_test",
    srcs = [
        "benchmark_tflite_model_test.cc",
    ],
    data = ["@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite"],
    tags = [
        "no_oss",  # TODO: b/361565588 - Re-enable.
    ],
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":benchmark_tflite_model_lib",
        "//tflite/core/c:c_api_types",
        "//tflite/tools:tool_params",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "benchmark_multirun_stats_recorder",
    hdrs = ["benchmark_multirun_stats_recorder.h"],
    copts = common_copts,
    deps = [":benchmark_model_lib"],
)

cc_library(
    name = "benchmark_performance_options",
    srcs = [
        "benchmark_performance_options.cc",
    ],
    hdrs = ["benchmark_performance_options.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_multirun_stats_recorder",
        ":benchmark_params",
        ":benchmark_utils",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/delegates/hexagon:hexagon_delegate",
        "//tflite/nnapi:nnapi_util",
        "//tflite/profiling:time",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/core/util:stats_calculator_portable",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "benchmark_params",
    hdrs = ["benchmark_params.h"],
    copts = common_copts,
    deps = ["//tflite/tools:tool_params"],
)

cc_library(
    name = "benchmark_model_lib",
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = ["benchmark_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        ":benchmark_utils",
        "//tflite:framework",
        "//tflite/core/c:common",
        "//tflite/profiling:memory_info",
        "//tflite/profiling:memory_usage_monitor",
        "//tflite/profiling:time",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "@org_tensorflow//tensorflow/core/util:stats_calculator_portable",
    ],
)

cc_library(
    name = "register_custom_op",
    srcs = [
        "register_custom_op.cc",
    ],
    hdrs = [
        "register_custom_op.h",
    ],
    copts = common_copts,
    deps = [
        "//tflite:op_resolver",
        "@com_google_absl//absl/base:core_headers",
    ],
    alwayslink = 1,
)

exports_files(["register_custom_op.h"])

cc_library(
    name = "benchmark_utils",
    srcs = [
        "benchmark_utils.cc",
    ],
    hdrs = ["benchmark_utils.h"],
    copts = common_copts,
    deps = ["//tflite/profiling:time"],
)

cc_test(
    name = "benchmark_utils_test",
    srcs = [
        "benchmark_utils_test.cc",
    ],
    deps = [
        ":benchmark_utils",
        "//tflite/profiling:time",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
