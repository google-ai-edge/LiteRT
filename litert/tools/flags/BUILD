# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "common_flags",
    srcs = ["common_flags.cc"],
    hdrs = ["common_flags.h"],
    deps = ["@com_google_absl//absl/flags:flag"],
    alwayslink = 1,
)

cc_library(
    name = "apply_plugin_flags",
    srcs = ["apply_plugin_flags.cc"],
    hdrs = ["apply_plugin_flags.h"],
    deps = [
        ":flag_types",
        "//litert/c/options:litert_compiler_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/options:compiler_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,
)

cc_test(
    name = "apply_plugin_flags_test",
    srcs = ["apply_plugin_flags_test.cc"],
    deps = [
        ":apply_plugin_flags",
        "//litert/c/options:litert_compiler_options",
        "//litert/cc:litert_expected",
        "//litert/cc/options:compiler_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flag_types",
    srcs = ["flag_types.cc"],
    hdrs = ["flag_types.h"],
    deps = [
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "flag_types_test",
    srcs = ["flag_types_test.cc"],
    deps = [
        ":flag_types",
        "@com_google_googletest//:gtest_main",
    ],
)
