load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:ios.bzl",
    "ios_application",
    "ios_extension",
    "ios_framework",
    "ios_static_framework",
    "ios_ui_test",
    "ios_unit_test",
)
load(
    "//apple:macos.bzl",
    "macos_application",
    "macos_framework",
    "macos_static_framework",
    # Enable when macos_test_runner supports macOS UI Tests.
    # "macos_ui_test",
    "macos_unit_test",
)
load(
    "//apple:tvos.bzl",
    "tvos_application",
    "tvos_extension",
    "tvos_static_framework",
)
load(
    "//apple:watchos.bzl",
    "watchos_application",
    "watchos_extension",
)

licenses(["notice"])

# iOS

## Code

swift_library(
    name = "ButtonsLib",
    srcs = [
        "Buttons/AppDelegate.swift",
        "Buttons/ViewController.swift",
    ],
    data = ["Buttons/Base.lproj/Main.storyboard"],
    module_name = "Buttons",
    tags = ["manual"],
    deps = [
        ":ButtonsShared2Lib",
        ":ButtonsSharedLib",
    ],
)

swift_library(
    name = "ButtonsExtensionLib",
    srcs = [
        "ButtonsExtension/ActionRequestHandler.swift",
    ],
    module_name = "ButtonsExtension",
    tags = ["manual"],
    deps = [
        ":ButtonsShared2Lib",
        ":ButtonsSharedLib",
    ],
)

swift_library(
    name = "ButtonsSharedLib",
    srcs = [
        "ButtonsShared/SomethingShared.swift",
    ],
    module_name = "ButtonsShared",
    tags = ["manual"],
)

swift_library(
    name = "ButtonsShared2Lib",
    srcs = [
        "ButtonsShared2/SomethingElseShared.swift",
    ],
    module_name = "ButtonsShared2",
    tags = ["manual"],
)

swift_library(
    name = "ButtonsTestsLib",
    testonly = True,
    srcs = [
        "ButtonsTests/ButtonsTests.swift",
    ],
    module_name = "ButtonsTests",
    tags = ["manual"],
    deps = [
        ":ButtonsLib",
    ],
)

swift_library(
    name = "ButtonsUITestsLib",
    testonly = True,
    srcs = [
        "ButtonsUITests/ButtonsUITests.swift",
    ],
    module_name = "ButtonsUITests",
    tags = ["manual"],
)

## Packaging

ios_framework(
    name = "ButtonsSharedFramework",
    bundle_id = "com.google.ButtonsSharedFramework",
    extension_safe = True,
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["ButtonsSharedFramework/Info.plist"],
    minimum_os_version = "11.0",
    deps = [
        ":ButtonsShared2Lib",
        ":ButtonsSharedLib",
    ],
)

ios_application(
    name = "Buttons",
    bundle_id = "com.google.Buttons",
    extensions = [":ButtonsExtension"],
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":ButtonsSharedFramework"],
    infoplists = ["Buttons/Info.plist"],
    minimum_os_version = "11.0",
    watch_application = ":ButtonsWatch",
    deps = [":ButtonsLib"],
)

ios_extension(
    name = "ButtonsExtension",
    bundle_id = "com.google.Buttons.ButtonsExtension",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":ButtonsSharedFramework"],
    infoplists = ["ButtonsExtension/Info.plist"],
    minimum_os_version = "11.0",
    deps = [":ButtonsExtensionLib"],
)

ios_static_framework(
    name = "ButtonsStaticFramework",
    bundle_name = "ButtonsShared",
    minimum_os_version = "9.0",
    deps = [":ButtonsSharedLib"],
)

## Tests

ios_unit_test(
    name = "ButtonsTests",
    minimum_os_version = "11.0",
    test_host = ":Buttons",
    deps = [":ButtonsTestsLib"],
)

ios_unit_test(
    name = "ButtonsLogicTests",
    minimum_os_version = "11.0",
    deps = [":ButtonsTestsLib"],
)

ios_ui_test(
    name = "ButtonsUITests",
    minimum_os_version = "11.0",
    test_host = ":Buttons",
    deps = [":ButtonsUITestsLib"],
)

test_suite(
    name = "iOSButtonsTestSuite",
    tests = [
        ":ButtonsLogicTests",
        ":ButtonsTests",
        ":ButtonsUITests",
    ],
)

# watchOS

## Code

swift_library(
    name = "ButtonsWatchExtensionLib",
    srcs = [
        "ButtonsWatchExtension/ComplicationController.swift",
        "ButtonsWatchExtension/ExtensionDelegate.swift",
        "ButtonsWatchExtension/InterfaceController.swift",
        "ButtonsWatchExtension/NotificationController.swift",
    ],
    module_name = "ButtonsWatchExtension",
    tags = ["manual"],
)

## Packaging

watchos_application(
    name = "ButtonsWatch",
    bundle_id = "com.google.Buttons.watchkitapp",
    extension = ":ButtonsWatchExtension",
    infoplists = ["ButtonsWatch/Info.plist"],
    minimum_os_version = "4.0",
    resources = ["ButtonsWatch/Base.lproj/Interface.storyboard"],
)

watchos_extension(
    name = "ButtonsWatchExtension",
    bundle_id = "com.google.Buttons.watchkitapp.watchkitextension",
    infoplists = ["ButtonsWatchExtension/Info.plist"],
    minimum_os_version = "4.0",
    deps = [
        ":ButtonsWatchExtensionLib",
    ],
)

# tvOS

## Code

swift_library(
    name = "ButtonsTVLib",
    srcs = [
        "ButtonsTV/AppDelegate.swift",
        "ButtonsTV/ViewController.swift",
    ],
    module_name = "ButtonsTV",
    tags = ["manual"],
)

swift_library(
    name = "ButtonsTVExtensionLib",
    srcs = [
        "ButtonsTVExtension/ServiceProvider.swift",
    ],
    module_name = "ButtonsTVExtension",
    tags = ["manual"],
)

swift_library(
    name = "ButtonsTVTestsLib",
    testonly = True,
    srcs = [
        "ButtonsTVTests/ButtonsTVTests.swift",
    ],
    tags = ["manual"],
)

swift_library(
    name = "ButtonsTVUITestsLib",
    testonly = True,
    srcs = [
        "ButtonsTVUITests/ButtonsTVUITests.swift",
    ],
    tags = ["manual"],
)

## Resources

objc_library(
    name = "ButtonsTVResources",
    data = ["ButtonsTV/Base.lproj/Main.storyboard"],
)

## Packaging

tvos_application(
    name = "ButtonsTV",
    bundle_id = "com.google.ButtonsTV",
    extensions = [":ButtonsTVExtension"],
    infoplists = ["ButtonsTV/Info.plist"],
    minimum_os_version = "11.0",
    deps = [
        ":ButtonsTVLib",
        ":ButtonsTVResources",
    ],
)

tvos_extension(
    name = "ButtonsTVExtension",
    bundle_id = "com.google.ButtonsTV.ButtonsTVExtension",
    infoplists = ["ButtonsTVExtension/Info.plist"],
    minimum_os_version = "11.0",
    deps = [":ButtonsTVExtensionLib"],
)

tvos_static_framework(
    name = "ButtonsTVStaticFramework",
    bundle_name = "ButtonsTV",
    minimum_os_version = "11.0",
    deps = [":ButtonsTVLib"],
)

# macOS

## Code

swift_library(
    name = "ButtonsMacLib",
    srcs = [
        "ButtonsMac/AppDelegate.swift",
        "ButtonsMac/ViewController.swift",
    ],
    module_name = "ButtonsMac",
    tags = ["manual"],
)

swift_library(
    name = "ButtonsMacTestsLib",
    testonly = True,
    srcs = [
        "ButtonsMacTests/ButtonsMacTests.swift",
    ],
    module_name = "ButtonsMacTests",
    tags = ["manual"],
    deps = [
        ":ButtonsMacLib",
    ],
)

swift_library(
    name = "ButtonsMacUITestsLib",
    testonly = True,
    srcs = [
        "ButtonsMacUITests/ButtonsMacUITests.swift",
    ],
    module_name = "ButtonsMacUITests",
    tags = ["manual"],
)

macos_static_framework(
    name = "ButtonsMacStaticFramework",
    bundle_name = "ButtonsMac",
    minimum_os_version = "11.11",
    deps = [":ButtonsMacLib"],
)

## Resources

objc_library(
    name = "ButtonsMacResources",
    data = ["ButtonsMac/Base.lproj/Main.storyboard"],
)

## Packaging

macos_framework(
    name = "ButtonsMacDynamicFramework",
    bundle_id = "com.google.ButtonsMac",
    bundle_name = "ButtonsMac",
    infoplists = ["ButtonsMac/Info.plist"],
    minimum_os_version = "11.11",
    deps = [":ButtonsMacLib"],
)

macos_application(
    name = "ButtonsMac",
    bundle_id = "com.google.ButtonsMac",
    infoplists = ["ButtonsMac/Info.plist"],
    minimum_os_version = "12.0",
    deps = [
        ":ButtonsMacLib",
        ":ButtonsMacResources",
    ],
)

## Tests

macos_unit_test(
    name = "ButtonsMacLogicTests",
    bundle_id = "com.google.logic",
    minimum_os_version = "12.0",
    tags = ["manual"],  # https://github.com/bazelbuild/continuous-integration/issues/1273
    deps = [":ButtonsMacTestsLib"],
)

macos_unit_test(
    name = "ButtonsMacTests",
    minimum_os_version = "12.0",
    tags = ["manual"],
    test_host = ":ButtonsMac",
    deps = [":ButtonsMacTestsLib"],
)

# Enable when macos_test_runner supports macOS UI Tests.
# macos_ui_test(
#     name = "ButtonsMacUITests",
#     minimum_os_version = "10.13",
#     test_host = ":ButtonsMac",
#     deps = [":ButtonsMacUITestsLib"],
# )

test_suite(
    name = "macOSButtonsTestSuite",
    tests = [
        # ":ButtonsMacLogicTests",
        # ":ButtonsMacTests",
        # ":ButtonsMacUITests",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [
        ":Buttons",
        ":ButtonsStaticFramework",
        ":ButtonsExtension",
        ":ButtonsMac",
        ":ButtonsMacStaticFramework",
        ":ButtonsMacDynamicFramework",
        ":ButtonsTV",
        ":ButtonsTVExtension",
        ":ButtonsTVStaticFramework",
        ":ButtonsWatch",
        ":ButtonsWatchExtension",
    ],
)
