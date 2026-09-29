# The headers are downloaded from the pinned TensorFlow release. See
# //third_party:tensorflow_shim.bzl.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "stats_calculator_portable",
    hdrs = [
        "stat_summarizer_options.h",
        "stats_calculator.h",
    ],
    deps = ["@xla//xla/tsl/util:stats_calculator_portable"],
)
