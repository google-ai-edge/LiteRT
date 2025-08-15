load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

cc_binary(
    name = "option_writer_generator",
    srcs = ["option_writer_generator.cc"],
    deps = [
        "//tflite/schema:schema_fbs_with_reflection",
        "@flatbuffers",
    ],
)

cc_library(
    name = "writer_lib_enum",
    hdrs = ["enum_mapping.h"],
    visibility = ["//platforms/darwinn/tflite/serialization:__subpackages__"],
    deps = [
        "//tflite:builtin_op_data",
        "//tflite/converter/schema:schema_fbs_with_mutable",
    ],
)

cc_library(
    name = "writer_lib",
    srcs = [
        "writer_lib.cc",
    ],
    hdrs = [
        "writer_lib.h",
    ],
    data = [
        ":option_writer_gen",
    ],
    textual_hdrs = ["option_writer_generated.h"],
    deps = [
        "//tflite:builtin_op_data",
        "//tflite:framework",
        "//tflite:schema_fbs_version",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/converter/tools/versioning",
        "//tflite/core:framework_stable",
        "//tflite/core:model_builder",
        "//tflite/core/c:common",
        "//tflite/schema:schema_conversion_utils",
        "//tflite/tools/serialization:writer_lib_enum",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_binary(
    name = "writer",
    srcs = ["writer.cc"],
    deps = [
        ":writer_lib",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":writer_lib",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/core/kernels:builtin_ops",
    ],
)

cc_test(
    name = "writer_lib_test",
    size = "small",
    srcs = ["writer_lib_test.cc"],
    deps = [
        ":writer_lib",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:util",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/core:framework",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels:subgraph_test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:logging",
    ],
)

genrule(
    name = "option_writer_gen",
    outs = ["option_writer_generated.h"],
    cmd = "$(location :option_writer_generator) $(@)",
    tools = [":option_writer_generator"],
)
