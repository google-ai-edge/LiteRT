load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/support_lib/metadata:build_defs.bzl", "stamp_metadata_parser_version")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["@org_tensorflow//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

stamp_metadata_parser_version(
    name = "metadata_parser_h",
    srcs = ["metadata_parser.h.template"],
    outs = ["metadata_parser.h"],
)

cc_library(
    name = "metadata_extractor",
    srcs = ["metadata_extractor.cc"],
    hdrs = ["metadata_extractor.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/support_lib/metadata:metadata_schema_cc",
        "//litert/support_lib/metadata/cc/utils:zip_readonly_mem_file",
        "//tflite/schema:schema_fbs",
        "//third_party/minizip:zlib_minizip",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "@org_tensorflow//tensorflow_lite_support/cc:common",
        "@org_tensorflow//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow//tensorflow_lite_support/cc/port:statusor",
    ],
)

cc_library(
    name = "metadata_version",
    srcs = ["metadata_version.cc"],
    hdrs = [
        "metadata_version.h",
        ":metadata_parser_h",
    ],
    deps = [
        "//litert/support_lib/metadata:metadata_schema_cc",
        "//tflite/c:common",
        "//tflite/kernels/internal:compatibility",
        "//tflite/tools:logging",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_library(
    name = "metadata_populator",
    srcs = ["metadata_populator.cc"],
    hdrs = ["metadata_populator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/support_lib/metadata:metadata_schema_cc",
        "//litert/support_lib/metadata/cc/utils:zip_writable_mem_file",
        "//tflite/schema:schema_fbs",
        "//third_party/minizip:zlib_minizip",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow_lite_support/cc:common",
        "@org_tensorflow//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow//tensorflow_lite_support/cc/port:statusor",
    ],
)
