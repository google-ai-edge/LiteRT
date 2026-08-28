load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["@org_tensorflow//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_pywrap_flatbuffers",
    srcs = [
        "flatbuffers_lib.cc",
    ],
    enable_stub_generation = True,
    features = ["-use_header_modules"],
    module_name = "_pywrap_flatbuffers",
    pytype_srcs = [
        "_pywrap_flatbuffers.pyi",
    ],
    deps = [
        "@flatbuffers",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)
