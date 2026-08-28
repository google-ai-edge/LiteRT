load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//litert/python:pytype.bzl", "pytype_strict_library")
load("//litert/support_lib/metadata:build_defs.bzl", "stamp_metadata_parser_version")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],  # Apache 2.0
)

stamp_metadata_parser_version(
    name = "metadata_parser_py",
    srcs = ["metadata_parser.py.template"],
    outs = ["metadata_parser.py"],
)

py_library(
    name = "metadata",
    srcs = [
        "metadata.py",
        ":metadata_parser_py",
    ],
    data = ["//litert/support_lib/metadata:metadata_schema.fbs"],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/support_lib/metadata:metadata_schema_py",
        "//litert/support_lib/metadata:schema_py",
        "//litert/support_lib/metadata/cc/python:_pywrap_metadata_version",
        "//litert/support_lib/metadata/flatbuffers_lib:_pywrap_flatbuffers",
        "@flatbuffers//:runtime_py",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "metadata_displayer",
    srcs = ["metadata_displayer.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":metadata",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "metadata_writer_for_task",
    srcs = ["metadata_writer_for_task.py"],
    visibility = [
        "@org_tensorflow//tensorflow_lite_support:users",
    ],
    deps = [
        ":metadata",
        "//litert/support_lib/metadata:metadata_schema_py",
        "//litert/support_lib/metadata/python/metadata_writers:metadata_info",
        "//litert/support_lib/metadata/python/metadata_writers:metadata_writer",
        "//litert/support_lib/metadata/python/metadata_writers:writer_utils",
    ],
)
