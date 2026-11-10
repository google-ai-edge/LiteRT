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

# Tests for the `TensorFlowLite` Swift module.
#
# These live in their own package so that `TensorFlowLite_Swift_AllDelegates`
# does not share an output directory with
# `//litert/swift:TensorFlowLite_Swift`. Both declare
# `module_name = "TensorFlowLite"`, so in a single package they would collide
# over `TensorFlowLite.swiftmodule` whenever both are analyzed in the same
# configuration.

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

# An all-delegates build of the wrapper, so that the tests can exercise the
# CoreML and Metal delegate types unconditionally.
swift_library(
    name = "TensorFlowLite_Swift_AllDelegates",
    testonly = 1,
    srcs = ["//litert/swift:TensorFlowLite_Swift_srcs"],
    features = ["swift.enable_testing"],
    linkopts = [
        "-Wl,-weak_framework,CoreML",
        "-Wl,-weak_framework,Metal",
    ],
    module_name = "TensorFlowLite",
    # Needs the Apple SDK, so it cannot be analyzed in the default
    # configuration that wildcard patterns use. It is still covered by
    # `:Tests`, which transitions it to an iOS platform.
    tags = [
        "apple",
        "manual",
    ],
    deps = [
        # Dependency placeholder: apple_frameworks Darwin,
        # Dependency placeholder: apple_frameworks Foundation,
        "//litert/swift:TensorFlowLiteC_objc",
        "//tflite/delegates/coreml:coreml_delegate",
        "//tflite/delegates/gpu:metal_delegate",
    ],
)

# The tests locate their models with `Bundle(for:).path(forResource:)`, so the
# test data has to be bundled as resources rather than passed as `data`.
objc_library(
    name = "Resources",
    testonly = 1,
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/add_quantized.bin",
        "//tflite:testdata/multi_add.bin",
        "//tflite:testdata/multi_signatures.bin",
    ],
    tags = ["apple"],
)

swift_library(
    name = "TestsLibrary",
    testonly = 1,
    srcs = glob(["*.swift"]),
    # These tests predate Swift 6 and use non-Sendable static fixtures, which
    # trip strict concurrency checking. Keep them in Swift 5 language mode so
    # they remain identical to the upstream TFLite sources.
    features = ["-swift.enable_v6"],
    # Apple-only; reached through the `:Tests` platform transition.
    tags = [
        "apple",
        "manual",
    ],
    deps = [
        ":Resources",
        ":TensorFlowLite_Swift_AllDelegates",
        # Dependency placeholder: apple_frameworks XCTest,
    ],
)

ios_unit_test(
    name = "Tests",
    size = "small",
    timeout = "moderate",
    minimum_os_version = "15.0",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_ordered_runner",
    tags = [
        "apple",
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":TestsLibrary",
    ],
)
