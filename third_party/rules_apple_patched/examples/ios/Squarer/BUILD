load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//apple:ios.bzl", "ios_unit_test")

licenses(["notice"])

objc_library(
    name = "SquarerLib",
    srcs = ["Sources/Squarer.m"],
    hdrs = ["Sources/Squarer.h"],
)

objc_library(
    name = "SquarerTestsLib",
    testonly = True,
    srcs = ["Tests/SquarerTests.m"],
    deps = [
        ":SquarerLib",
    ],
)

ios_unit_test(
    name = "SquarerTests",
    env = {"TEST_ENV_VAR": "test_value"},
    minimum_os_version = "11.0",
    deps = [":SquarerTestsLib"],
)

ios_unit_test(
    name = "SquarerTestsOrdered",
    env = {"TEST_ENV_VAR": "test_value"},
    minimum_os_version = "11.0",
    runner = "//apple/testing/default_runner:ios_xctestrun_ordered_runner",
    deps = [":SquarerTestsLib"],
)

ios_unit_test(
    name = "SquarerTestsRandom",
    env = {"TEST_ENV_VAR": "test_value"},
    minimum_os_version = "11.0",
    runner = "//apple/testing/default_runner:ios_xctestrun_random_runner",
    deps = [":SquarerTestsLib"],
)
