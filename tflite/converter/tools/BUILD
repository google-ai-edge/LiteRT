load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite/converter:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(command_line_flags)

cc_library(
    name = "command_line_flags",
    srcs = ["command_line_flags.cc"],
    hdrs = ["command_line_flags.h"],
    copts = tflite_copts(),
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

# LINT.ThenChange(//tflite/tools:command_line_flags)
