load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:string_util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

cc_library(
    name = "interpreter_wrapper_lib",
    srcs = ["interpreter_wrapper.cc"],
    hdrs = [
        "interpreter_wrapper.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":numpy",
        ":python_error_reporter",
        ":python_utils",
        "//tflite:framework",
        "//tflite:shared_library",
        "//tflite:string_util",
        "//tflite:util",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/utils/experimental/stable_delegate:delegate_loader",
        "//tflite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/kernels:reference_ops",
        "//tflite/kernels/internal:compatibility",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:stateful_error_reporter",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

config_setting(
    name = "tflite_pip_with_flex",
    define_values = {
        "tflite_pip_with_flex": "true",
    },
)

pybind_extension(
    name = "_pywrap_tensorflow_interpreter_wrapper",
    srcs = [
        "interpreter_wrapper_pybind11.cc",
    ],
    hdrs = ["interpreter_wrapper.h"],
    additional_stubgen_deps = [
        "@org_tensorflow//third_party/py/numpy:numpy",
    ],
    common_lib_packages = [
        "litert/python",
        "tflite/python",
    ],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_tensorflow_interpreter_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":interpreter_wrapper_lib",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "@local_xla//third_party/python_runtime:headers",
        "@org_tensorflow//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
    ] + select({
        ":tflite_pip_with_flex": ["//tflite/delegates/flex:delegate"],
        "//conditions:default": [],
    }),
)
