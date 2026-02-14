load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//test:configurations.bzl",
    "IOS_64BIT_SIMULATOR_FAT_DEVICE_CONFIGURATIONS",
    "IOS_CONFIGURATIONS",
    "MACOS_CONFIGURATIONS",
    "TVOS_CONFIGURATIONS",
    "WATCHOS_CONFIGURATIONS",
)
load(
    "//test:test_rules.bzl",
    "apple_multi_shell_test",
    "apple_shell_test",
)
load(
    "//test/starlark_tests:common.bzl",
    "common",
)

licenses(["notice"])

bzl_library(
    name = "configurations",
    srcs = ["configurations.bzl"],
)

bzl_library(
    name = "test_rules",
    srcs = ["test_rules.bzl"],
)

# Export shell test utilities so can can be pulled in as data to the tests.
exports_files(["apple_shell_testutils.sh"])

# List of dependencies to be used in Starlark verifier tests.
filegroup(
    name = "apple_verification_test_deps",
    testonly = True,
    srcs = [
        "apple_shell_testutils.sh",
        "unittest.bash",
    ],
    # This is required as a default attribute of apple_verification_test but
    # should be considered private for other uses
    visibility = ["//visibility:public"],
)

# ----------------------------------------------------------------------------

# Ballpark number for shared_count:
#  grep -c function\ test test/*_test.sh

# Tests that end up needing things like ibtool or actool run under the hood
# are better off as a dedicated *_resources_test.sh that can then be marked
# with `flaky = True`. When building for some Apple platforms, the tools can end
# up talking to other services running on the machine (CoreSimulator) during
# the build and flake/fail. So by grouping them and marking them as flaky, it
# allows bazel to auto retry them and help minimize the unreliability in those
# tools.

apple_shell_test(
    name = "apple_bundle_version_test",
    size = "medium",
    src = "apple_bundle_version_test.sh",
)

apple_shell_test(
    name = "apple_core_ml_library_package_test",
    size = "medium",
    src = "apple_core_ml_library_package_test.sh",
    data = [
        "//test/testdata/resources:mlmodel_srcs",
    ],
)

apple_shell_test(
    name = "apple_core_ml_resource_test",
    size = "medium",
    src = "apple_core_ml_resource_test.sh",
    data = [
        "//test/testdata/resources:mlmodel_srcs",
    ],
)

apple_shell_test(
    name = "apple_intent_library_test",
    size = "medium",
    src = "apple_intent_library_test.sh",
    data = [
        "//test/testdata/resources:intent_srcs",
    ],
)

apple_shell_test(
    name = "apple_metal_library_test",
    size = "medium",
    src = "apple_metal_library_test.sh",
    data = [
        "//test/testdata/resources:metal_files",
    ],
)

apple_multi_shell_test(
    name = "ios_application_test",
    size = "large",
    src = "ios_application_test.sh",
    configurations = IOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "ios_application_resources_test",
    size = "large",
    src = "ios_application_resources_test.sh",
    configurations = IOS_CONFIGURATIONS,
    data = [
        "//test/testdata/resources:resource_data_deps_ios",
        "//test/testdata/resources:resource_data_deps_platform_independent",
        "//test/testdata/resources:texture_atlas_data_deps",
    ],
    flaky = True,
)

apple_multi_shell_test(
    name = "ios_application_swift_test",
    size = "large",
    src = "ios_application_swift_test.sh",
    configurations = IOS_64BIT_SIMULATOR_FAT_DEVICE_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "ios_extension_test",
    size = "medium",
    src = "ios_extension_test.sh",
    configurations = IOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "ios_imessage_test",
    size = "medium",
    src = "ios_imessage_test.sh",
    configurations = IOS_CONFIGURATIONS,
    data = [
        "//test/testdata/resources:resource_data_deps_ios",
    ],
)

apple_multi_shell_test(
    name = "macos_application_test",
    size = "medium",
    src = "macos_application_test.sh",
    configurations = MACOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "macos_application_resources_test",
    size = "medium",
    src = "macos_application_resources_test.sh",
    configurations = MACOS_CONFIGURATIONS,
    data = [
        "//test/testdata/resources:resource_data_deps_platform_independent",
    ],
)

apple_multi_shell_test(
    name = "macos_bundle_test",
    size = "medium",
    src = "macos_bundle_test.sh",
    configurations = MACOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "macos_quick_look_plugin_test",
    size = "medium",
    src = "macos_quick_look_plugin_test.sh",
    configurations = MACOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "tvos_application_test",
    size = "medium",
    src = "tvos_application_test.sh",
    configurations = TVOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "tvos_application_swift_test",
    size = "medium",
    src = "tvos_application_swift_test.sh",
    configurations = TVOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "tvos_extension_test",
    size = "medium",
    src = "tvos_extension_test.sh",
    configurations = TVOS_CONFIGURATIONS,
)

apple_multi_shell_test(
    name = "watchos_application_test",
    size = "medium",
    src = "watchos_application_test.sh",
    configurations = WATCHOS_CONFIGURATIONS,
)

apple_shell_test(
    name = "ios_coverage_test",
    size = "large",
    src = "ios_coverage_test.sh",
    tags = ["exclusive"],
)

apple_shell_test(
    name = "macos_test_runner_unit_test",
    size = "large",
    src = "macos_test_runner_unit_test.sh",
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "ios_xctestrun_runner_unit_test",
    size = "large",
    src = "ios_xctestrun_runner_unit_test.sh",
    args = [
        "--ios_multi_cpus=sim_arm64,x86_64",
    ],
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "ios_xctestrun_runner_ui_test",
    size = "large",
    src = "ios_xctestrun_runner_ui_test.sh",
    args = [
        "--ios_multi_cpus=sim_arm64,x86_64",
    ],
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "ios_test_runner_unit_test",
    size = "large",
    src = "ios_test_runner_unit_test.sh",
    args = [
        "--ios_multi_cpus=sim_arm64,x86_64",
    ],
    flaky = True,
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "ios_test_runner_unit_test_14.x",
    size = "large",
    src = "ios_test_runner_unit_test_14.x.sh",
    args = [
        "--ios_multi_cpus=sim_arm64,x86_64",
    ],
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "ios_xctest_swift_testing_runner_unit_test_17.x",
    size = "large",
    src = "ios_xctest_swift_testing_runner_unit_test_17.x.sh",
    args = [
        "--ios_multi_cpus=sim_arm64,x86_64",
    ],
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "ios_test_runner_ui_test",
    size = "large",
    src = "ios_test_runner_ui_test.sh",
    args = [
        "--ios_multi_cpus=sim_arm64,x86_64",
    ],
    tags = common.skip_ci_tags,
)

apple_shell_test(
    name = "smart_resource_deduplication_test",
    size = "large",
    src = "smart_resource_deduplication_test.sh",
    data = [
        "//test/testdata/resources:resource_data_deps_ios",
        "//test/testdata/resources:resource_data_deps_platform_independent",
    ],
)

# apple_shell_test(
#     name = "docc_archive_test",
#     src = "docc_archive_test.sh",
# )
