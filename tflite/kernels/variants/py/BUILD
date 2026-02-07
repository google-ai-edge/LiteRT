load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

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

py_test(
    name = "end_to_end_test",
    srcs = ["end_to_end_test.py"],
    strict_deps = True,
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
