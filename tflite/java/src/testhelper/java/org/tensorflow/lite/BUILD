# Description:
# Internal helper function to test TF Lite API.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_java//java:defs.bzl", "java_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

android_library(
    name = "testhelper",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//tflite/java:tensorflowlite_java",
    ],
)

java_library(
    name = "testhelper_javalib",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//tflite/java:tensorflowlite_javalib",
    ],
)
