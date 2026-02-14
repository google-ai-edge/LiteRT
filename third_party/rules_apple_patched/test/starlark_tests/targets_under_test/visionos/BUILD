# TODO(nglevin): describe this package.

load(
    "//apple:visionos.bzl",
    "visionos_application",
)
load(
    "//test/starlark_tests:common.bzl",
    "common",
)

licenses(["notice"])

package(
    default_testonly = True,
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

visionos_application(
    name = "app_minimal",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_visionos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_visionos.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

visionos_application(
    name = "app",
    app_icons = ["//test/starlark_tests/resources:VisionAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_visionos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_visionos.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:empty_strings_file_lib",
        "//test/starlark_tests/resources:nested_bundle_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

visionos_application(
    name = "swift_app",
    app_icons = ["//test/starlark_tests/resources:VisionAppIcon.xcassets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_visionos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)
