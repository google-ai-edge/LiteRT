load("//litert/python:pytype.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "metadata_writer",
    srcs = [
        "metadata_writer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":writer_utils",
        "//litert/support_lib/metadata:metadata_schema_py",
        "//litert/support_lib/metadata:schema_py",
        "//litert/support_lib/metadata/python:metadata",
        "@flatbuffers//:runtime_py",
    ],
)

pytype_strict_library(
    name = "metadata_info",
    srcs = [
        "metadata_info.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":writer_utils",
        "//litert/support_lib/metadata:metadata_schema_py",
        "//litert/support_lib/metadata:schema_py",
    ],
)

pytype_strict_library(
    name = "writer_utils",
    srcs = [
        "writer_utils.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/support_lib/metadata:metadata_schema_py",
        "//litert/support_lib/metadata:schema_py",
    ],
)

pytype_strict_library(
    name = "image_classifier",
    srcs = [
        "image_classifier.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//litert/support_lib/metadata:metadata_schema_py",
    ],
)

pytype_strict_library(
    name = "object_detector",
    srcs = [
        "object_detector.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//litert/support_lib/metadata:metadata_schema_py",
        "//litert/support_lib/metadata:schema_py",
        "//litert/support_lib/metadata/python:metadata",
        "@flatbuffers//:runtime_py",
    ],
)

pytype_strict_library(
    name = "image_segmenter",
    srcs = [
        "image_segmenter.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//litert/support_lib/metadata:metadata_schema_py",
    ],
)

pytype_strict_library(
    name = "nl_classifier",
    srcs = [
        "nl_classifier.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
    ],
)

pytype_strict_library(
    name = "audio_classifier",
    srcs = [
        "audio_classifier.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
    ],
)

pytype_strict_library(
    name = "bert_nl_classifier",
    srcs = [
        "bert_nl_classifier.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
    ],
)
