load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load(
    "//apple:macos.bzl",
    "macos_application",
)
load(
    "//apple:versioning.bzl",
    "apple_bundle_version",
)

licenses(["notice"])

swift_library(
    name = "Sources",
    srcs = ["Sources/AppDelegate.swift"],
    data = [
        "Resources/Main.storyboard",
    ],
)

apple_bundle_version(
    name = "HelloWorldSwiftVersion",
    build_version = "1.0",
)

macos_application(
    name = "HelloWorldSwift",
    app_icons = ["//examples/resources:MacAppIcon.xcassets"],
    bundle_id = "com.example.hello-world-swift",
    infoplists = [":Info.plist"],
    minimum_os_version = "10.13",
    version = ":HelloWorldSwiftVersion",
    deps = [":Sources"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [":HelloWorldSwift"],
)
