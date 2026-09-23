load("@flatbuffers//:build_defs.bzl", "flatbuffer_android_library", "flatbuffer_cc_library", "flatbuffer_java_library", "flatbuffer_py_library")
load("//tflite/java:aar_with_jni.bzl", "aar_without_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["metadata_schema.fbs"])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tflite/converter/schema:schema.fbs"],
)

# Generic schema for model metadata.
flatbuffer_cc_library(
    name = "metadata_schema_cc",
    srcs = ["metadata_schema.fbs"],
    compatible_with = ["//buildenv/target:non_prod"],
)

flatbuffer_py_library(
    name = "metadata_schema_py",
    srcs = ["metadata_schema.fbs"],
)

flatbuffer_java_library(
    name = "metadata_schema_java",
    srcs = ["metadata_schema.fbs"],
    custom_package = "com.google.ai.edge.litert.support.metadata.schema",
)

flatbuffer_android_library(
    name = "metadata_schema_fbs_android",
    srcs = ["metadata_schema.fbs"],
    custom_package = "com.google.ai.edge.litert.support.metadata.schema",
)

# Maven AAR for the LiteRT metadata library. It bundles the generated
# FlatBuffers bindings, and doesn't depend on the LiteRT runtime.
#
# bazel build --config=android_arm64 --android_ndk_min_sdk_version=24 \
#   //litert/support_lib/metadata:litert-metadata-aar
aar_without_jni(
    name = "litert-metadata-aar",
    android_library = "//litert/support_lib/metadata/java:litert-metadata-android-lib",
)
