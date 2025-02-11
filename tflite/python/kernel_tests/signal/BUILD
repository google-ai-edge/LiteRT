load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//tflite/python:interpreter",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/eager:def_function",
    ],
)

cuda_py_strict_test(
    name = "window_ops_test",
    srcs = ["window_ops_test.py"],
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":test_util",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:for_generated_wrappers",
        "@org_tensorflow//tensorflow/python/framework:tensor_spec",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/signal:window_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//third_party/py/numpy",
    ],
)
