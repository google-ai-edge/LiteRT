load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = ["//visibility:private"],
)

filegroup(
    name = "toco_python_api_hdrs",
    srcs = [
        "toco_python_api.h",
    ],
    visibility = [
        "@org_tensorflow//tensorflow/python:__subpackages__",
    ],
)

cc_library(
    name = "toco_python_api",
    srcs = ["toco_python_api.cc"],
    hdrs = ["toco_python_api.h"],
    features = ["-parse_headers"],
    visibility = [
        "@org_tensorflow//tensorflow/python:__subpackages__",
    ],
    deps = [
        "//tflite:model_builder",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/python/interpreter_wrapper:python_error_reporter",
        "//tflite/python/interpreter_wrapper:python_utils",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:model_flags_proto_cc",
        "//tflite/toco:toco_convert",
        "//tflite/toco:toco_flags_proto_cc",
        "//tflite/toco:toco_graphviz_dump_options",
        "//tflite/toco:toco_port",
        "//tflite/toco:toco_tooling",
        "//tflite/toco:tooling_util",
        "//tflite/toco:types_proto_cc",
        "//tflite/toco/logging:conversion_log_util",
        "//tflite/toco/logging:toco_conversion_log_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf_headers",
        "@flatbuffers//:runtime_cc",
        "@local_tsl//tsl/platform:status",
        "@local_xla//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@org_tensorflow//tensorflow/c:kernels",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ] + select({
        # This is required when running `tflite_convert` from `bazel`.
        # It requires to link with TensorFlow Ops to get the op definitions.
        ":tflite_convert_with_select_tf_ops": [
            "@org_tensorflow//tensorflow/core:ops",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

# Compatibility stub. Remove when internal customers moved.
py_strict_library(
    name = "tensorflow_wrap_toco",
    srcs = ["tensorflow_wrap_toco.py"],
    visibility = ["//learning/expander/pod/deep_pod/utils:__subpackages__"],
    deps = [
        "@org_tensorflow//tensorflow/python:_pywrap_toco_api",
    ],
)
