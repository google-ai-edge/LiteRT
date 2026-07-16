# Copyright 2026 Google LLC.
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

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")

LITERT_MIN_IOS_VERSION = "15.0"

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

objc_library(
    name = "litert_objc_environment",
    srcs = [
        "sources/LRTEnvironment.mm",
        "sources/LRTError.mm",
    ],
    hdrs = [
        "apis/LRTEnvironment.h",
        "apis/LRTError.h",
    ],
    deps = [
        "//third_party/apple_frameworks:Foundation",
        # Dependency placeholder: apple_frameworks Metal,
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
    ],
)

objc_library(
    name = "litert_objc_environment_test_library",
    testonly = True,
    srcs = [
        "tests/LRTEnvironmentTests.mm",
    ],
    deps = [
        ":litert_objc_environment",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

ios_unit_test(
    name = "litert_objc_environment_test_ios",
    minimum_os_version = LITERT_MIN_IOS_VERSION,
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":litert_objc_environment_test_library",
    ],
)

objc_library(
    name = "litert_objc_options",
    srcs = [
        "sources/LRTOptions.mm",
    ],
    hdrs = [
        "apis/LRTOptions.h",
    ],
    deps = [
        "//litert/cc:litert_options",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "litert_objc_options_test_library",
    testonly = True,
    srcs = [
        "tests/LRTOptionsTests.mm",
    ],
    deps = [
        ":litert_objc_options",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

ios_unit_test(
    name = "litert_objc_options_test_ios",
    minimum_os_version = LITERT_MIN_IOS_VERSION,
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":litert_objc_options_test_library",
    ],
)
