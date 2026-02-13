load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load(
    "//apple:macos.bzl",
    "macos_application",
    "macos_xpc_service",
)
load(
    "//apple:versioning.bzl",
    "apple_bundle_version",
)

licenses(["notice"])

swift_library(
    name = "XPCServiceAppLib",
    srcs = [
        "XPCServiceAppSources/AppDelegate.swift",
        "XPCServiceAppSources/ViewController.swift",
        "XPCServiceSources/ServiceProtocol.swift",
    ],
    data = ["XPCServiceAppSources/Main.storyboard"],
)

swift_library(
    name = "XPCServiceLib",
    srcs = [
        "XPCServiceSources/ServiceProtocol.swift",
        "XPCServiceSources/main.swift",
    ],
)

apple_bundle_version(
    name = "XPCServiceAppVersion",
    build_version = "1.0",
)

macos_xpc_service(
    name = "XPCService",
    bundle_id = "com.example.xpc-service-app.xpc-service",
    infoplists = ["XPCServiceSources/Info.plist"],
    minimum_os_version = "10.13",
    version = ":XPCServiceAppVersion",
    deps = [":XPCServiceLib"],
)

macos_application(
    name = "XPCServiceApp",
    app_icons = ["//examples/resources:MacAppIcon.xcassets"],
    bundle_id = "com.example.xpc-service-app",
    infoplists = ["XPCServiceAppSources/Info.plist"],
    minimum_os_version = "10.13",
    version = ":XPCServiceAppVersion",
    xpc_services = [":XPCService"],
    deps = [":XPCServiceAppLib"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [":XPCServiceApp"],
)
