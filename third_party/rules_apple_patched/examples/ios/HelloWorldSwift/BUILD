load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("//apple:docc.bzl", "docc_archive")
load("//apple:ios.bzl", "ios_application", "ios_ui_test", "ios_unit_test")
load("//apple:xctrunner.bzl", "xctrunner")

licenses(["notice"])

swift_library(
    name = "Sources",
    srcs = [
        "Sources/BazelApp.swift",
    ],
    data = [
        "Resources/HelloWorldSwift.docc",
        "//examples/resources:Localizable.xcstrings",
    ],
    module_name = "Sources",
    tags = ["manual"],
)

swift_library(
    name = "Tests",
    testonly = True,
    srcs = [
        "Tests/Tests.swift",
    ],
    tags = ["manual"],
    deps = [
        ":Sources",
    ],
)

# Pass the depending UI test's name as module_name to support test_filter
swift_library(
    name = "UITests",
    testonly = True,
    srcs = [
        "Tests/UITests.swift",
    ],
    module_name = "HelloWorldSwiftUITests",
    tags = ["manual"],
)

ios_unit_test(
    name = "UnitTests",
    minimum_os_version = "15.0",
    deps = [":Tests"],
)

genrule(
    name = "CreateTestCoverageManifest",
    srcs = ["Sources/BazelApp.swift"],
    outs = [
        "CoverageManifest.instrumented_files",
    ],
    cmd = "echo $(SRCS) > $@",
)

ios_unit_test(
    name = "UnitTestsWithCoverageManifest",
    minimum_os_version = "15.0",
    test_coverage_manifest = "CoverageManifest.instrumented_files",
    deps = [":Tests"],
)

ios_application(
    name = "HelloWorldSwift",
    app_icons = ["//examples/resources:PhoneAppIcon.xcassets"],
    bundle_id = "com.example.hello-world-swift",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [":Info.plist"],
    minimum_os_version = "15.0",
    deps = [":Sources"],
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [":HelloWorldSwift"],
)

ios_ui_test(
    name = "HelloWorldSwiftUITests",
    minimum_os_version = "15.0",
    runner = "//apple/testing/default_runner:ios_xctestrun_ordered_runner",
    test_host = ":HelloWorldSwift",
    deps = [":UITests"],
)

# Example using docc_archive to generate documentation
docc_archive(
    name = "HelloWorldSwift.doccarchive",
    default_code_listing_language = "en",
    dep = ":HelloWorldSwift",
    diagnostic_level = "information",
    emit_extension_block_symbols = "1",
    enable_inherited_docs = True,
    fallback_bundle_identifier = "com.example.hello-world-swift",
    fallback_bundle_version = "1.0.0",
    fallback_display_name = "HelloWorldSwift",
    minimum_access_level = "internal",
)

xctrunner(
    name = "HelloWorldSwiftXCTRunner",
    testonly = True,
    tests = [":HelloWorldSwiftUITests"],
)
