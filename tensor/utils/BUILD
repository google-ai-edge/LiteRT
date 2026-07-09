load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    includes = ["../utils"],  # copybara:comment
    deps = [
        ":source_location",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "macros_test",
    srcs = ["macros_test.cc"],
    deps = [
        ":macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        ":macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

# This should mirror the :matchers target and is provided for compatibility
# with internal targets that are built exclusively for OSS.
cc_library(
    name = "matchers_no_g3",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        ":macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
)

# OSS: source_location_cc_library
