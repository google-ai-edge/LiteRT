load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite/core:special_rules.bzl", "delegate_registry_visibility_allowlist")
load("//tflite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_registry",
    hdrs = ["delegate_registry.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = delegate_registry_visibility_allowlist(),
    deps = [
        "//tflite/core/acceleration/configuration:delegate_registry",
    ],
)

cc_library(
    name = "stable_delegate_registry",
    hdrs = ["stable_delegate_registry.h"],
    visibility = experimental_acceleration_api_allowlist(),
    deps = [
        "//tflite/core/acceleration/configuration:stable_delegate_registry",
    ],
)
