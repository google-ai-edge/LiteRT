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

# Pin the released Maven artifact to Java 17 bytecode (class file major 61),
# matching the "litert-support-api" AAR.
# See //litert/support_lib/BUILD for why "-source 17 -target 17"
# is used instead of "--release 17".
JAVACOPTS = select({
    "//litert/support_lib:public_maven_build": ["-source 17 -target 17"],
    "//conditions:default": [],
})

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

# The target backing the "litert-metadata" Maven AAR, which includes the
# metadata Android library, the metadata schema Java binding, and the TFLite
# schema Java binding.
#
# The bindings have to be compiled in, rather than pulled in through "deps":
# "aar_without_jni" packages only the given library's own .jar, and the
# generated bindings aren't published as Maven artifacts.
android_library(
    name = "litert-metadata-android-lib",
    srcs = METADATA_SRCS + [
        "//litert/support_lib/metadata:metadata_schema_fbs_android_srcjar",
        "//litert/support_lib/metadata/schema:schema_fbs_android_srcjar",
    ],
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        # Need to be consistent as the deps used in flatbuffer_android_library.
        "@flatbuffers//:runtime_android",
        "@org_checkerframework_qual",
        "@litert_maven//:com_google_errorprone_error_prone_annotations",
    ],
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
    deps = [
        # Need to be consistent as the deps used in flatbuffer_java_library.
        "@flatbuffers//:runtime_java",
        "@org_checkerframework_qual",
        "@litert_maven//:com_google_errorprone_error_prone_annotations",
    ],
)
