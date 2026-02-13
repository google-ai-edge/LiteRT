load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load(
    "//apple:macos.bzl",
    "macos_command_line_application",
)
load(
    "//apple:versioning.bzl",
    "apple_bundle_version",
)

licenses(["notice"])

swift_library(
    name = "Sources",
    srcs = ["Sources/main.swift"],
)

apple_bundle_version(
    name = "CommandLineSwiftVersion",
    build_version = "1.0",
)

macos_command_line_application(
    name = "CommandLineSwift",
    bundle_id = "com.example.command-line-swift",
    infoplists = [":Info.plist"],
    minimum_os_version = "10.13",
    version = ":CommandLineSwiftVersion",
    deps = [":Sources"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [":CommandLineSwift"],
)
