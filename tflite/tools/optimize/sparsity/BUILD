load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "format_converter_wrapper_pybind11",
    srcs = ["format_converter_wrapper_pybind11.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    data = [
        "format_converter_wrapper_pybind11.pyi",
    ],
    enable_stub_generation = True,
    features = ["-use_header_modules"],
    deps = [
        "//tflite/core/c:common",
        "//tflite/kernels/internal/utils:sparsity_format_converter",
        "@org_tensorflow//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_strict_test(
    name = "format_converter_wrapper_pybind11_test",
    srcs = ["format_converter_wrapper_pybind11_test.py"],
    python_version = "PY3",
    deps = [
        ":format_converter_wrapper_pybind11",
        "@absl_py//absl/testing:absltest",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
    ],
)
