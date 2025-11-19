# Custom Ops useful for GenAI models.
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

cc_library(
    name = "genai_ops",
    srcs = [
        "external_kvcache.cc",
        "genai_ops.cc",
        "kvcache.cc",
        "sdpa.cc",
    ],
    hdrs = [
        "genai_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:subgraph",
        "//tflite/core/c:common",
        "//tflite/experimental/resource",
        "//tflite/experimental/resource:cache_buffer",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:common",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "kvcache_test",
    srcs = ["kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":genai_ops",
        "//tflite/c:c_api_types",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "external_kvcache_test",
    srcs = ["external_kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":genai_ops",
        "//tflite:framework",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "pywrap_genai_ops",
    srcs = [
        "genai_ops_wrapper.cc",
    ],
    hdrs = ["genai_ops.h"],
    additional_exported_symbols = ["GenAIOpsRegisterer"],
    common_lib_packages = [
        "litert/python",
        "tflite/python",
    ],
    enable_stub_generation = True,
    link_in_framework = True,
    module_name = "pywrap_genai_ops",
    pytype_srcs = [
        "pywrap_genai_ops.pyi",
    ],
    visibility = ["//visibility:public"],
    wrap_py_init = True,
    deps = [
        ":genai_ops",
        "//tflite:framework_stable",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tflite_portable_test_suite()
