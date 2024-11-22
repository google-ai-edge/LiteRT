# Description:
# Java Native Interface (JNI) library for testing the TensorFlow Lite Java API.

load(
    "//tflite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite",
    "jni_binary_with_tflite",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "native_srcs",
    srcs = [
        "interpreter_test_jni.cc",
        "nnapi_delegate_test_jni.cc",
        "supported_features_jni.cc",
        "test_init_jni.cc",
    ],
)

cc_library_with_tflite(
    name = "native",
    testonly = 1,
    srcs = [
        "interpreter_test_jni.cc",
        "nnapi_delegate_test_jni.cc",
        "supported_features_jni.cc",
    ],
    tflite_deps = [
        ":test_init_jni",
        "//tflite/delegates/nnapi/java/src/main/native",
        "//tflite/java/src/main/native",
        "//tflite/java/src/main/native:jni_utils",
        "//tflite/java/src/main/native:native_framework_only",
    ],
    deps = [
        "//tflite/core/c:common",
        "//tflite/delegates/nnapi:nnapi_delegate",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/java/jni",
        "//tflite/kernels:kernel_util",
        "//tflite/nnapi:nnapi_lib",
        "//tflite/nnapi/sl:nnapi_support_library_headers",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "test_init_jni",
    testonly = 1,
    srcs = [
        "test_init_jni.cc",
    ],
    tflite_deps = [
        "//tflite/java/src/main/native:jni_utils",
        "//tflite/c:test_util",
    ],
    deps = [
        "//tflite/java/jni",
    ],
    alwayslink = 1,
)

# Same as "native", but excluding dependencies on experimental features.
cc_library_with_tflite(
    name = "native_stable",
    testonly = 1,
    srcs = [
        "supported_features_jni.cc",
        "test_init_jni.cc",
    ],
    tflite_deps = [
        "//tflite/c:test_util",
        "//tflite/delegates/nnapi/java/src/main/native",
        "//tflite/java/src/main/native:jni_utils",
        "//tflite/java/src/main/native:native_stable",
        "//tflite/java/src/main/native:native_stable_framework_only",
    ],
    deps = [
        "//tflite/core/c:common",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/java/jni",
        "//tflite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

jni_binary_with_tflite(
    name = "libtensorflowlite_test_jni.so",
    testonly = 1,
    linkscript = "//tflite/java:tflite_version_script.lds",
    tflite_deps = [":native"],
)

jni_binary_with_tflite(
    name = "libtensorflowlite_stable_test_jni.so",
    testonly = 1,
    linkscript = "//tflite/java:tflite_version_script.lds",
    tflite_deps = [":native_stable"],
)

# Dummy native library which doesn't actually contain the TFLite implementation.
jni_binary_with_tflite(
    name = "libtensorflowlite_jni.so",
    testonly = 1,
)
