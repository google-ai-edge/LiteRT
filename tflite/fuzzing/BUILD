# Fuzzing harnesses for the TensorFlow Lite runtime.

load(
    "@org_tensorflow//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "interpreter_fuzz",
    srcs = ["interpreter_fuzz.cc"],
    deps = [
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/kernels:builtin_ops",
    ],
)
