load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "operator",
    srcs = [
        "operator.cc",
    ],
    hdrs = [
        "builtin_operator.h",
        "custom_operator.h",
        "operator.h",
        "simple_operator.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite/toco:__subpackages__",
    ],
    deps = [
        ":types",
        "//tflite/c:c_api_types",
        "//tflite/converter/delegates/flex:allowlisted_flex_ops_lib",
        "//tflite/converter/tools/versioning",
        "//tflite/converter/tools/versioning:op_signature",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:graph_transformations",
        "//tflite/toco:model",
        "//tflite/toco:runtime",
        "//tflite/toco:toco_port",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "operator_test",
    srcs = [
        "operator_test.cc",
    ],
    deps = [
        ":operator",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:runtime",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:ops",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "types",
    srcs = [
        "types.cc",
    ],
    hdrs = [
        "types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:string_util",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:runtime",
        "@com_google_absl//absl/log",
        "@flatbuffers//:runtime_cc",
    ],
)

tf_cc_test(
    name = "types_test",
    srcs = [
        "types_test.cc",
    ],
    deps = [
        ":types",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:runtime",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/core:ops",
    ],
)

cc_library(
    name = "export",
    srcs = [
        "export.cc",
    ],
    hdrs = [
        "export.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":operator",
        ":types",
        "//tflite:schema_fbs_version",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/converter/quantization/lite/toco_legacy:quantize_weights",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/converter/tools/versioning",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:toco_port",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:lib_proto_parsing",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

tf_cc_test(
    name = "export_test",
    srcs = [
        "export_test.cc",
    ],
    deps = [
        ":export",
        ":operator",
        ":types",
        "//tflite/converter/schema:schema_utils",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@org_tensorflow//tensorflow/core:lib_proto_parsing",
        "@org_tensorflow//tensorflow/core:ops",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "import",
    srcs = [
        "import.cc",
    ],
    hdrs = [
        "import.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":operator",
        ":types",
        "//tflite:framework",
        "//tflite/converter/schema:schema_utils",
        "//tflite/core:framework",
        "//tflite/core/tools:verifier",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:model_flags_proto_cc",
        "//tflite/toco:tooling_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "import_test",
    srcs = [
        "import_test.cc",
    ],
    tags = [
        "no_oss",  # TODO(b/273558651): Enable after updating flatbuffer version.
    ],
    deps = [
        ":import",
        "//tflite:schema_fbs_version",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/schema:schema_fbs",
        "//tflite/toco:model",
        "//tflite/toco:model_flags_proto_cc",
        "//tflite/toco:toco_port",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:ops",
    ],
)
