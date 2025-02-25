load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

pybind_extension(
    name = "register_list_ops_py",
    srcs = ["register_list_ops_py.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "register_list_ops_py.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite:mutable_op_resolver",
        "//tflite/kernels/variants:list_ops_lib",
        "@pybind11",
    ],
)

py_strict_test(
    name = "end_to_end_test",
    srcs = ["end_to_end_test.py"],
    tags = ["nochromiumos_arm"],
    deps = [
        ":register_list_ops_py",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:interpreter",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/ops:list_ops",
        "@org_tensorflow//tensorflow/python/platform:test",
    ],
)
