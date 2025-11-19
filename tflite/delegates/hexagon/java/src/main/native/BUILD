# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Hexagon delegate Java API using the TensorFlow Lite CC library.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/delegates/hexagon/java:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "native",
    srcs = ["hexagon_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite/c:c_api_types",
        "//tflite/delegates/hexagon:hexagon_delegate",
        "//tflite/java/jni",
    ],
    alwayslink = 1,
)
