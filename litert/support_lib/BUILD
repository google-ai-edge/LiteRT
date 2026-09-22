# Description:
# LiteRT Support API in Java.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tflite/java:aar_with_jni.bzl", "aar_without_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "AndroidManifest.xml",
    "AndroidManifestImpl.xml",
])

filegroup(
    name = "java_srcs",
    srcs = glob(["java/com/google/ai/edge/litert/support/**/*.java"]),
)

# Android Library target for LiteRT Support Library. It depends on the
# LiteRT Runtime (tensorflow/lite/java:tensorflowlite),
# including its associated native code. If you don't want to introduce the native
# library into dependencies, use "litert_support_java" instead,
# which depends on tensorflow/lite/java:tensorflowlite_api.
android_library(
    name = "litert_support",
    srcs = [],
    custom_package = "com.google.ai.edge.litert.support.impl",
    manifest = "AndroidManifestImpl.xml",
    exports = [
        ":litert_support_java",
    ],
    deps = [
        ":litert_support_java",
        "//tflite/java:tensorflowlite_stable",
    ],
)

# Android Library target for LiteRT Support Library, excluding native code,
# and excluding dependencies on the LiteRT Runtime.
# If you use this target, a LiteRT Runtime (and it's associated
# native code) is nevertheless still required, and must be provided by a
# separate dependency.
android_library(
    name = "litert_support_java",
    srcs = [":java_srcs"],
    custom_package = "com.google.ai.edge.litert.support",
    manifest = "AndroidManifest.xml",
    # LINT.IfChange(dep)
    deps = [
        "//tflite/java:tensorflowlite_api",
        "@com_google_auto_value",
        "@litert_maven//:com_google_android_odml_image",
        "@litert_maven//:com_google_errorprone_error_prone_annotations",
        "@maven//:androidx_annotation_annotation",
        "@org_checkerframework_qual",
    ],
    # LINT.ThenChange(//ci/build_android_package.sh)
)

# This alias matches the associated .aar library name output style.
alias(
    name = "litert-support",
    actual = ":litert_support",
)

# Maven AAR containing the LiteRT Support Java API. It doesn't bundle the LiteRT
# runtime; the "litert-api" Maven artifact provides it.
#
# bazel build --config=android_arm64 --android_ndk_min_sdk_version=24 \
#   //litert/support_lib:litert-support-api-aar
aar_without_jni(
    name = "litert-support-api-aar",
    android_library = ":litert_support_java",
)

# Maven AAR for the LiteRT Support Library. The AAR itself is an aggregator: its
# POM supplies the "litert-support-api" and "litert" Maven artifacts.
#
# bazel build --config=android_arm64 --android_ndk_min_sdk_version=24 \
#   //litert/support_lib:litert-support-aar
aar_without_jni(
    name = "litert-support-aar",
    android_library = ":litert_support",
)
