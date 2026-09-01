# Description:
# LiteRT Support API in Java for metadata.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_java//java:defs.bzl", "java_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

METADATA_SRCS = glob(
    ["com/google/ai/edge/litert/support/metadata/**/*.java"],
)

android_library(
    name = "litert_metadata",
    srcs = METADATA_SRCS,
    manifest = "AndroidManifest.xml",
    deps = [
        "//litert/support_lib/metadata:metadata_schema_fbs_android",
        "//litert/support_lib/metadata/schema:schema_fbs_android",
        "@litert_maven//:com_google_errorprone_error_prone_annotations",
        "@org_checkerframework_qual",
    ],
)

alias(
    name = "litert-metadata",
    actual = ":litert_metadata",
)

java_library(
    name = "litert_metadata_lib",
    srcs = METADATA_SRCS,
    javacopts = ["--release 8"],
    deps = [
        "//litert/support_lib/metadata:metadata_schema_java",
        "//litert/support_lib/metadata/schema:schema_fbs_java",
        "@litert_maven//:com_google_errorprone_error_prone_annotations",
        "@org_checkerframework_qual",
    ],
)

# The target for OSS release, which includes the metadata Java library, the
# metadata schema Java binding, and the TFLite schema Java binding.
java_library(
    name = "litert-metadata-lib",
    srcs = METADATA_SRCS + [
        "//litert/support_lib/metadata:metadata_schema_java_srcjar",
        "//litert/support_lib/metadata/schema:schema_fbs_java_srcjar",
    ],
    javacopts = ["--release 8"],
    # LINT.IfChange(dep)
    deps = [
        # Need to be consistent as the deps used in flatbuffer_java_library.
        "@flatbuffers//:runtime_java",
        "@org_checkerframework_qual",
        "@litert_maven//:com_google_errorprone_error_prone_annotations",
    ],
    # LINT.ThenChange(//ci/build_android_package.sh)
)
