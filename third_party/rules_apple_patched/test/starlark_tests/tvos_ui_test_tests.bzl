# Copyright 2019 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tvos_ui_test Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:apple_dsym_bundle_info_test.bzl",
    "apple_dsym_bundle_info_test",
)
load(
    "//test/starlark_tests/rules:apple_verification_test.bzl",
    "apple_verification_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def tvos_ui_test_test_suite(name):
    """Test suite for tvos_ui_test.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test",
        output_group_name = "dsyms",
        expected_outputs = [
            "app.app.dSYM/Contents/Info.plist",
            "app.app.dSYM/Contents/Resources/DWARF/app",
            "ui_test.xctest.dSYM/Contents/Info.plist",
            "ui_test.xctest.dSYM/Contents/Resources/DWARF/ui_test",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_apple_dsym_bundle_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test",
        expected_direct_dsyms = ["dSYMs/ui_test.xctest.dSYM"],
        expected_transitive_dsyms = ["dSYMs/app.app.dSYM", "dSYMs/ui_test.xctest.dSYM"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "ui_test",
            "CFBundleIdentifier": "com.google.exampleTests",
            "CFBundleName": "ui_test",
            "CFBundlePackageType": "BNDL",
            "CFBundleSupportedPlatforms:0": "AppleTV*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "appletvsimulator",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "appletvsimulator*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_tvos.baseline,
            "UIDeviceFamily:0": "3",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_test_bundle_id_override".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test_custom_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "my.test.bundle.id",
        },
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_test_bundle_id_same_as_test_host_error".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test_invalid_bundle_id",
        expected_error = "The test bundle's identifier of 'com.google.example' can't be the same as the test host's bundle identifier. Please change one of them.",
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_base_bundle_id_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ui_test_with_base_bundle_id_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.ui-test-with-base-bundle-id-derived-bundle-id",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
