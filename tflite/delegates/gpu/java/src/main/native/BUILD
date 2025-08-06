# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite GPU delegate Java API using the TensorFlow Lite CC library.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "gpu_delegate_jni_srcs",
    srcs = [
        "gpu_delegate_jni.cc",
        "gpu_delegate_native_jni.cc",
    ],
)

cc_library_with_tflite(
    name = "compatibility_list_jni",
    srcs = ["compatibility_list_jni.cc"],
    copts = tflite_copts(),
    tflite_deps = [
        "//tflite/java/src/main/native:jni_utils",
    ],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/gpu/gl:request_gpu_info",
        "//tflite/experimental/acceleration/compatibility:android_info",
        "//tflite/experimental/acceleration/compatibility:gpu_compatibility",
        "//tflite/java/jni",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "gpu_delegate_jni",
    srcs = ["gpu_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    tflite_deps = [
        "//tflite/java/src/main/native:jni_utils",
    ],
    deps = [
        "//tflite/delegates/gpu:delegate",
        "//tflite/delegates/gpu:delegate_options",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/gpu/gl:request_gpu_info",
        "//tflite/experimental/acceleration/compatibility:android_info",
        "//tflite/experimental/acceleration/compatibility:gpu_compatibility",
        "//tflite/java/jni",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gpu_delegate_native_jni",
    srcs = ["gpu_delegate_native_jni.cc"],
    copts = tflite_copts(),
    deps = [
        "//tflite/java/jni",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "native",
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-lGLESv3",
            "-lEGL",
        ],
        "//conditions:default": [],
    }),
    tflite_deps = [
        ":compatibility_list_jni",
        ":gpu_delegate_jni",
    ],
    deps = [
        ":gpu_delegate_native_jni",
    ],
)
