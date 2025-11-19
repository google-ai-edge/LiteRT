load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "if_portable", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "metrics_wrapper_lib",
    srcs = if_portable(
        if_false = ["wrapper/metrics_wrapper_nonportable.cc"],
        if_true = ["wrapper/metrics_wrapper_portable.cc"],
    ),
    hdrs = ["wrapper/metrics_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "@local_xla//third_party/python_runtime:headers",
    ] + if_portable(
        if_false = [
            "//learning/brain/google/monitoring:metrics_exporter",
        ],
        if_true = [],
    ),
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_metrics_wrapper",
    srcs = ["wrapper/metrics_wrapper_pybind11.cc"],
    hdrs = ["wrapper/metrics_wrapper.h"],
    common_lib_packages = [
        "litert/python",
        "tflite/python",
    ],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tensorflow_lite_metrics_wrapper.pyi",
    ],
    visibility = ["//visibility:public"],
    wrap_py_init = True,
    deps = [
        ":metrics_wrapper_lib",
        "@com_google_protobuf//:protobuf",
        "@local_xla//third_party/python_runtime:headers",
        "@org_tensorflow//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "metrics_wrapper",
    srcs = ["wrapper/metrics_wrapper.py"],
    deps = [
        ":_pywrap_tensorflow_lite_metrics_wrapper",
        "//tflite/converter/metrics:converter_error_data_proto_py",
        "//tflite/converter/python:wrap_converter",
    ],
)

py_strict_test(
    name = "metrics_wrapper_test",
    srcs = ["wrapper/metrics_wrapper_test.py"],
    deps = [
        ":metrics_wrapper",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:convert",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "metrics_interface",
    srcs = ["metrics_interface.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

genrule(
    name = "metrics_py_gen",
    srcs = if_portable(
        if_false = ["metrics_nonportable.py"],
        if_true = ["metrics_portable.py"],
    ),
    outs = ["metrics.py"],
    cmd = (
        "cat $(SRCS) > $(OUTS)"
    ),
    compatible_with = get_compatible_with_portable(),
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tflite:__subpackages__"],
    deps = if_portable(
        if_false = [
            "//tflite/converter/metrics:converter_error_data_proto_py",
            ":metrics_wrapper",
            "@org_tensorflow//tensorflow/python/eager:monitoring",
        ],
        if_true = [],
    ) + [":metrics_interface"],
)

py_strict_test(
    name = "metrics_test",
    srcs = if_portable(
        if_false = ["metrics_nonportable_test.py"],
        if_true = ["metrics_portable_test.py"],
    ),
    data = [
        "//tflite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    main = if_portable(
        if_false = "metrics_nonportable_test.py",
        if_true = "metrics_portable_test.py",
    ),
    tags = ["notap"],  # TODO(b/373657707): Remove once we debug the failure.
    deps = [
        ":metrics",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/converter/metrics:converter_error_data_proto_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "//tflite/python:convert",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/eager:context",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:convert_to_constants",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:importer",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
    ],
)
