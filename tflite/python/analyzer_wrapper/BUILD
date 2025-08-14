load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pybind_extension(
    name = "_pywrap_analyzer_wrapper",
    srcs = [
        "analyzer_wrapper.cc",
    ],
    common_lib_packages = [
        "litert/python",
        "tflite/python",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_analyzer_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":model_analyzer",
        "@pybind11",
    ],
)

cc_library(
    name = "model_analyzer",
    srcs = ["model_analyzer.cc"],
    hdrs = ["model_analyzer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite:version",
        "//tflite/converter/schema:schema_utils",
        "//tflite/core:model_builder",
        "//tflite/core/api:error_reporter",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/public:release_version",
    ],
)
