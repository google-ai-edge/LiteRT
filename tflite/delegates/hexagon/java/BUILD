load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_jni_binary", "tflite_jni_linkopts")
load("//tflite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# EXPERIMENTAL: Native target that runs inference on the Hexagon backend.
# The Hexagon backend-related targets are intentionally not included in BUILD.bazel.
tflite_jni_binary(
    name = "libtensorflowlite_hexagon_jni.so",
    linkopts = tflite_jni_linkopts() + select({
        "@org_tensorflow//tensorflow:android": ["-ldl"],
        "//conditions:default": [],
    }),
    linkscript = "//tflite/delegates/hexagon:version_script.lds",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite/delegates/hexagon/java/src/main/native",
    ],
)

cc_library(
    name = "tensorflowlite_hexagon",
    srcs = [
        "libtensorflowlite_hexagon_jni.so",
    ] + select({
        # copybara:uncomment_begin(separate repo in OSS)
        # "@org_tensorflow//tensorflow:android_arm64": ["//tflite/delegates/hexagon/hexagon_nn:libhexagon_interface.so"],
        # "@org_tensorflow//tensorflow:android_arm": ["//tflite/delegates/hexagon/hexagon_nn:libhexagon_interface.so"],
        # copybara:uncomment_end_and_comment_begin
        "@org_tensorflow//tensorflow:android_arm64": ["@hexagon_nn//:hexagon/arm64-v8a/libhexagon_interface.so"],
        "@org_tensorflow//tensorflow:android_arm": ["@hexagon_nn//:hexagon/armeabi-v7a/libhexagon_interface.so"],
        # copybara:comment_end
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
)

android_library(
    name = "tensorflowlite_java_hexagon",
    srcs = ["//tflite/delegates/hexagon/java/src/main/java/org/tensorflow/lite:hexagon_delegate"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":tensorflowlite_hexagon",
        "//tflite/java:tensorflowlite_java",
        "@org_checkerframework_qual",
    ],
)

aar_with_jni(
    name = "tensorflow-lite-hexagon",
    android_library = ":tensorflowlite_java_hexagon",
    headers = ["//tflite/delegates/hexagon:hexagon_delegate.h"],
)
