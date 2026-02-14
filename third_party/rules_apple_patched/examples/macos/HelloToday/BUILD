load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:macos.bzl",
    "macos_application",
    "macos_extension",
)
load(
    "//apple:versioning.bzl",
    "apple_bundle_version",
)

licenses(["notice"])

objc_library(
    name = "AppSources",
    srcs = [
        "AppSources/AppDelegate.h",
        "AppSources/AppDelegate.m",
        "AppSources/ViewController.h",
        "AppSources/ViewController.m",
        "AppSources/main.m",
    ],
    data = [
        "AppResources/Main.storyboard",
    ],
)

objc_library(
    name = "ExtSources",
    srcs = [
        "ExtSources/TodayViewController.h",
        "ExtSources/TodayViewController.m",
    ],
    data = [
        "ExtResources/TodayViewController.xib",
    ],
    sdk_frameworks = [
        "NotificationCenter",
    ],
)

apple_bundle_version(
    name = "HelloTodayVersion",
    build_version = "1.0",
)

macos_application(
    name = "HelloToday",
    app_icons = ["//examples/resources:MacAppIcon.xcassets"],
    bundle_id = "com.example.hello-today",
    extensions = [
        ":TodayExtension",
    ],
    infoplists = [":App-Info.plist"],
    minimum_os_version = "10.13",
    version = ":HelloTodayVersion",
    deps = [":AppSources"],
)

macos_extension(
    name = "TodayExtension",
    bundle_id = "com.example.hello-today.today-extension",
    entitlements = ":Ext-Entitlements.entitlements",
    infoplists = [":Ext-Info.plist"],
    minimum_os_version = "10.13",
    version = ":HelloTodayVersion",
    deps = [":ExtSources"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [
        ":HelloToday",
        ":TodayExtension",
    ],
)
