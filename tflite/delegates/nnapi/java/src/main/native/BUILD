# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite NNAPI Delegate Java API using the TensorFlow Lite CC library.

load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite:special_rules.bzl", "nnapi_native_srcs_visibility_allowlist")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "native",
    srcs = ["nnapi_delegate_impl_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tflite/delegates/nnapi:nnapi_delegate",
        "//tflite/java/jni",
    ],
    alwayslink = 1,
)

filegroup(
    name = "native_srcs",
    srcs = ["nnapi_delegate_impl_jni.cc"],
    visibility = nnapi_native_srcs_visibility_allowlist(),
)
