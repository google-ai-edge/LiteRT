load("@org_tensorflow//tensorflow:tensorflow.bzl", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension", "replace_with_portable_tf_lib_when_required")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

# Utilities for signature_defs in TFLite
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "signature_def_util",
    srcs = ["signature_def_util.cc"],
    hdrs = ["signature_def_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    features = select({
        "@org_tensorflow//tensorflow:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    deps = replace_with_portable_tf_lib_when_required([
        "@org_tensorflow//tensorflow/core:lib_proto_parsing",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ]) + [
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:protos_all_cc_impl",
    ],
)

cc_test(
    name = "signature_def_util_test",
    size = "small",
    srcs = ["signature_def_util_test.cc"],
    data = [
        "//tflite:testdata/add.bin",
    ],
    tags = [
        "no_oss",
        "tflite_not_portable",
    ],
    deps = [
        ":signature_def_util",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

pybind_extension(
    name = "_pywrap_signature_def_util_wrapper",
    srcs = [
        "signature_def_util_wrapper_pybind11.cc",
    ],
    data = [
        "_pywrap_signature_def_util_wrapper.pyi",
    ],
    enable_stub_generation = True,
    deps = [
        ":signature_def_util",
        "//tflite:framework",
        "//tflite/core:framework",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core/protobuf:for_core_protos_cc",
        "@org_tensorflow//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
    ],
)

py_library(
    name = "signature_def_utils",
    srcs = ["signature_def_utils.py"],
    strict_deps = True,
    deps = [
        ":_pywrap_signature_def_util_wrapper",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "signature_def_utils_test",
    srcs = ["signature_def_utils_test.py"],
    data = ["//tflite:testdata/add.bin"],
    strict_deps = True,
    tags = [
        "no_mac",
        "no_oss",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":signature_def_utils",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

tflite_portable_test_suite()
