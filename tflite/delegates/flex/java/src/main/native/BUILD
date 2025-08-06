# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Flex delegate for using TensorFlow ops with TensorFlow Lite.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(
    srcs = ["flex_delegate_jni.cc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "native",
    srcs = ["flex_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tflite/delegates/flex:delegate",
        "//tflite/delegates/utils:simple_delegate",
        "//tflite/java/jni",
        "//tflite/testing:init_tensorflow",
    ],
    alwayslink = 1,
)
