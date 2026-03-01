load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_binary(
    name = "modify_model_interface",
    srcs = ["modify_model_interface.py"],
    strict_deps = True,
    deps = [
        ":modify_model_interface_constants",
        ":modify_model_interface_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "modify_model_interface_lib",
    srcs = ["modify_model_interface_lib.py"],
    strict_deps = True,
    deps = [
        ":_pywrap_modify_model_interface",
        ":modify_model_interface_constants",
        "//tflite/python:schema_py",
    ],
)

# Use --config=disable_tf_lite_py when running this test under github.
py_test(
    name = "modify_model_interface_lib_test",
    srcs = ["modify_model_interface_lib_test.py"],
    strict_deps = True,
    deps = [
        ":modify_model_interface_lib",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "modify_model_interface_constants",
    srcs = ["modify_model_interface_constants.py"],
    strict_deps = True,
    deps = ["@org_tensorflow//tensorflow/python/framework:dtypes"],
)

pybind_extension(
    name = "_pywrap_modify_model_interface",
    srcs = ["modify_model_interface.cc"],
    common_lib_packages = [
        "litert/python",
        "tflite/python",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_modify_model_interface.pyi",
    ],
    wrap_py_init = True,
    deps = [
        "//tflite/schema:schema_fbs",
        "//tflite/tools/optimize:modify_model_interface",
        "@pybind11",
    ],
)
