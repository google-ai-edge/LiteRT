load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

common_copts = tflite_copts() + tflite_copts_warnings()

# Library of list ops.
cc_library(
    name = "list_ops_lib",
    srcs = glob(
        ["list_kernels/*.cc"],
        exclude = ["list_kernels/*test*"],
    ) + ["register_list_ops.cc"],
    hdrs = [
        "list_ops_lib.h",
        "register_list_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//tflite:array",
        "//tflite:mutable_op_resolver",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:cpu_backend_context",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:optimized_base",
        "//tflite/kernels/internal:runtime_shape",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/schema:schema_fbs",
    ],
)

# Utils for simple kernel unit tests.
cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["list_kernels/test_util.cc"],
    hdrs = ["list_kernels/test_util.h"],
    deps = [
        ":tensor_array",
        "//tflite:array",
        "//tflite:type_to_tflitetype",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:op_macros",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_reserve_test",
    srcs = ["list_kernels/list_reserve_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_stack_test",
    srcs = ["list_kernels/list_stack_test.cc"],
    deps = [
        ":list_ops_lib",
        ":test_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_set_item_test",
    srcs = ["list_kernels/list_set_item_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite:type_to_tflitetype",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_from_tensor_test",
    srcs = ["list_kernels/list_from_tensor_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_get_item_test",
    srcs = ["list_kernels/list_get_item_test.cc"],
    deps = [
        ":list_ops_lib",
        ":test_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_length_test",
    srcs = ["list_kernels/list_length_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_element_shape_test",
    srcs = ["list_kernels/list_element_shape_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_pop_back_test",
    srcs = ["list_kernels/list_pop_back_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_push_back_test",
    srcs = ["list_kernels/list_push_back_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite:type_to_tflitetype",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "variant_add_n_test",
    srcs = ["list_kernels/variant_add_n_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite:type_to_tflitetype",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "variant_zeros_like_test",
    srcs = ["list_kernels/variant_zeros_like_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "list_ops_util",
    srcs = ["list_ops_util.cc"],
    hdrs = ["list_ops_util.h"],
    visibility = ["//tflite/kernels/variants:__subpackages__"],
    deps = [
        ":tensor_array",
        "//tflite:array",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "list_ops_util_test",
    srcs = ["list_ops_util_test.cc"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//tflite:array",
        "//tflite:util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Utils for testing complex multi-op subgraphs.
cc_library(
    name = "list_ops_subgraph_test_util",
    testonly = True,
    srcs = ["list_ops_subgraph_test_util.cc"],
    hdrs = ["list_ops_subgraph_test_util.h"],
    deps = [
        ":list_ops_lib",
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/core:subgraph",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:op_macros",
        "//tflite/kernels:subgraph_test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "list_ops_subgraph_test",
    srcs = ["list_ops_subgraph_test.cc"],
    deps = [
        ":list_ops_subgraph_test_util",
        ":tensor_array",
        "//tflite:array",
        "//tflite:interpreter_test_util",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:op_macros",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# Type erased class that sits behind the `data.data` of variant tensors.
cc_library(
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "//tflite:array",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "tensor_array_test",
    srcs = ["tensor_array_test.cc"],
    deps = [
        ":tensor_array",
        "//tflite:array",
        "//tflite:type_to_tflitetype",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Benchmark tool override with registered custom tensorlist ops.
cc_library(
    name = "register_list_ops_for_benchmark",
    srcs = ["register_list_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_lib",
        "//tflite:mutable_op_resolver",
        "//tflite:op_resolver",
        "//tflite/tools/benchmark:register_custom_op",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_with_list_ops",
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
        ":register_list_ops_for_benchmark",
        "//tflite/tools/benchmark:benchmark_model_main",
    ],
)
