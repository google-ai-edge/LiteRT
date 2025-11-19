load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "@org_tensorflow//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "toco_conversion_log_proto",
    srcs = ["toco_conversion_log.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "conversion_log_util",
    srcs = ["conversion_log_util.cc"],
    hdrs = ["conversion_log_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":toco_conversion_log_proto_cc",
        "//tflite:version",
        "//tflite/toco:model",
        "//tflite/toco:tooling_util",
        "//tflite/toco/tflite:export",
        "//tflite/toco/tflite:operator",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "conversion_log_util_test",
    srcs = ["conversion_log_util_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_log_util",
        "//tflite/toco:model",
        "//tflite/toco:model_flags_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:ops",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

filegroup(
    name = "html_template",
    srcs = [
        "template.html",
    ],
)

py_strict_library(
    name = "gen_html",
    srcs = ["gen_html.py"],
    data = [
        "html_template",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":toco_conversion_log_proto_py",
        "@org_tensorflow//tensorflow/python/lib/io:file_io",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

py_strict_test(
    name = "gen_html_test",
    srcs = ["gen_html_test.py"],
    data = [
        ":template.html",
        "//tflite/toco/logging/testdata:generated.html",
        "//tflite/toco/logging/testdata:toco_log_after.pb",
        "//tflite/toco/logging/testdata:toco_log_before.pb",
        "//tflite/toco/logging/testdata:toco_tf_graph.dot",
        "//tflite/toco/logging/testdata:toco_tflite_graph.dot",
    ],
    deps = [
        ":gen_html",
        ":toco_conversion_log_proto_py",
        #internal proto upb dep
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/lib/io:file_io",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "toco_conversion_log_proto_py",
#     visibility = ["//visibility:public"],
#     deps = [":toco_conversion_log_proto"],
# )
# copybara:uncomment_end
