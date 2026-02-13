load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:ios.bzl",
    "ios_application",
)
load(
    "//apple:versioning.bzl",
    "apple_bundle_version",
)
load(
    "//apple:watchos.bzl",
    "watchos_application",
    "watchos_extension",
)

licenses(["notice"])

objc_library(
    name = "PhoneSources",
    srcs = [
        "PhoneSources/AppDelegate.h",
        "PhoneSources/AppDelegate.m",
        "PhoneSources/ViewController.h",
        "PhoneSources/ViewController.m",
        "PhoneSources/main.m",
    ],
    data = [
        "PhoneResources/Main.storyboard",
    ],
    sdk_frameworks = [
        "WatchConnectivity",
    ],
    tags = ["manual"],
)

objc_library(
    name = "WatchSharedSources",
    srcs = [
        "WatchSharedSources/InterfaceController.h",
        "WatchSharedSources/InterfaceController.m",
    ],
    sdk_frameworks = [
        "WatchConnectivity",
        "WatchKit",
    ],
    tags = ["manual"],
)

objc_library(
    name = "WatchApp2Sources",
    srcs = [
        "WatchApp2Sources/ExtensionDelegate.h",
        "WatchApp2Sources/ExtensionDelegate.m",
    ],
    tags = ["manual"],
    deps = [":WatchSharedSources"],
)

objc_library(
    name = "SingleTargetWatchSources",
    srcs = [
        "SingleTargetWatchSources/AppDelegate.h",
        "SingleTargetWatchSources/AppDelegate.m",
        "SingleTargetWatchSources/main.m",
    ],
    tags = ["manual"],
    deps = [":WatchSharedSources"],
)

apple_bundle_version(
    name = "HelloWorldVersion",
    build_version = "1.0",
)

ios_application(
    name = "HelloWorld",
    app_icons = ["//examples/resources:PhoneAppIcon.xcassets"],
    bundle_id = "com.example.hello-world",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [":Phone-Info.plist"],
    launch_storyboard = "//examples/resources:Launch.storyboard",
    minimum_os_version = "11.0",
    version = ":HelloWorldVersion",
    watch_application = ":HelloWorld-WatchApplication",
    deps = [":PhoneSources"],
)

watchos_application(
    name = "HelloWorld-WatchApplication",
    app_icons = ["//examples/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.example.hello-world.watch",
    extensions = [":HelloWorld-WatchExtension"],
    infoplists = [":WatchApp-Info.plist"],
    minimum_os_version = "4.0",
    storyboards = [
        "WatchResources/Interface.storyboard",
    ],
    version = ":HelloWorldVersion",
)

watchos_extension(
    name = "HelloWorld-WatchExtension",
    bundle_id = "com.example.hello-world.watch.extension",
    infoplists = [":WatchExt-Info.plist"],
    minimum_os_version = "4.0",
    version = ":HelloWorldVersion",
    deps = [":WatchApp2Sources"],
)

ios_application(
    name = "HelloWorld-SingleTargetWatchHost",
    app_icons = ["//examples/resources:PhoneAppIcon.xcassets"],
    bundle_id = "com.example.hello-world",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [":Phone-Info.plist"],
    launch_storyboard = "//examples/resources:Launch.storyboard",
    minimum_os_version = "11.0",
    version = ":HelloWorldVersion",
    watch_application = ":HelloWorld-SingleTargetWatchApplication",
    deps = [":PhoneSources"],
)

watchos_application(
    name = "HelloWorld-SingleTargetWatchApplication",
    app_icons = ["//examples/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.example.hello-world.watch",
    infoplists = [":WatchApp-Info.plist"],
    minimum_os_version = "7.0",
    storyboards = [
        "WatchResources/Interface.storyboard",
    ],
    version = ":HelloWorldVersion",
    deps = [":SingleTargetWatchSources"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [
        ":HelloWorld",
        ":HelloWorld-SingleTargetWatchHost",
        ":HelloWorld-SingleTargetWatchApplication",
        ":HelloWorld-WatchApplication",
        ":HelloWorld-WatchExtension",
    ],
)
