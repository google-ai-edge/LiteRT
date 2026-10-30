# The sources are downloaded from the pinned TensorFlow release. See
# //third_party:tensorflow_shim.bzl.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11_lib",
    hdrs = ["pybind11_lib.h"],
    features = ["-parse_headers"],
    deps = ["@pybind11"],
)
