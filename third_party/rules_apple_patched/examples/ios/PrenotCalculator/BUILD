# Copyright 2015 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:ios.bzl",
    "ios_application",
    "ios_unit_test",
)
load("//apple:resources.bzl", "apple_resource_bundle")

licenses(["notice"])

ios_application(
    name = "PrenotCalculator",
    bundle_id = "com.example.prenot-calculator",
    families = ["iphone"],
    infoplists = ["PrenotCalculator-Info.plist"],
    minimum_os_version = "11.0",
    deps = [":PrenotCalculator_library"],
)

objc_library(
    name = "PrenotCalculator_library",
    srcs = [
        "AppDelegate.m",
        "CalculatedValues.m",
        "CalculatorViewController.m",
        "CoreData.m",
        "Equation.m",
        "Expression.m",
        "Literal.m",
        "ValuesViewController.m",
        "main.m",
    ],
    hdrs = glob(["**/*.h"]),
    data = glob(["DataModel.xcdatamodeld/**"]) + [
        "CalculatorViewController.xib",
        ":PrenotCalculatorResources",
    ],
    sdk_frameworks = ["CoreData"],
    tags = ["manual"],
)

apple_resource_bundle(
    name = "PrenotCalculatorResources",
    resources = glob(["Resources/**"]),
)

objc_library(
    name = "PrenotCalculatorTestsLib",
    testonly = True,
    srcs = [
        "EquationTest.m",
    ],
    tags = ["manual"],
    deps = [
        ":PrenotCalculator_library",
    ],
)

ios_unit_test(
    name = "PrenotCalculatorTests",
    minimum_os_version = "11.0",
    deps = [":PrenotCalculatorTestsLib"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [":PrenotCalculator"],
)
