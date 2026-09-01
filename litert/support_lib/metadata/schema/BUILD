# Generic TFLite schema bindings used by the LiteRT metadata library.

load("@flatbuffers//:build_defs.bzl", "flatbuffer_android_library", "flatbuffer_java_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# Keep these generators in a separate package from the metadata schema
# generators. The FlatBuffers rules use package-scoped temporary directories,
# so colocating them can make concurrent builds overwrite each other's output.
flatbuffer_android_library(
    name = "schema_fbs_android",
    srcs = ["//tflite/converter/schema:schema.fbs"],
    custom_package = "org.tensorflow.lite.schema",
)

flatbuffer_java_library(
    name = "schema_fbs_java",
    srcs = ["//tflite/converter/schema:schema.fbs"],
    custom_package = "org.tensorflow.lite.schema",
)
