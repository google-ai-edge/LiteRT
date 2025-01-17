load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_analyzer_wrapper.pyi",
    ],
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
        "//tflite/core:model_builder",
        "//tflite/core/api:error_reporter",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/public:release_version",
    ],
)
