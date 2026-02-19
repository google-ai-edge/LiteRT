# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""watchos_ui_test Starlark tests leveraging watchos_application."""

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

def watchos_ui_test_test_suite(name):
    """Test suite for watchos_ui_test leveraging watchos_application.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ui_test",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ui_test",
        output_group_name = "dsyms",
        expected_outputs = [
            "ext.appex.dSYM/Contents/Info.plist",
            "ext.appex.dSYM/Contents/Resources/DWARF/ext",
            "ui_test.xctest.dSYM/Contents/Info.plist",
            "ui_test.xctest.dSYM/Contents/Resources/DWARF/ui_test",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_apple_dsym_bundle_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ui_test",
        expected_direct_dsyms = ["dSYMs/ui_test.xctest.dSYM"],
        expected_transitive_dsyms = ["dSYMs/ext.appex.dSYM", "dSYMs/ui_test.xctest.dSYM"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ui_test",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "ui_test",
            "CFBundleIdentifier": "com.google.exampleTests",
            "CFBundleName": "ui_test",
            "CFBundlePackageType": "BNDL",
            "CFBundleSupportedPlatforms:0": "WatchSimulator*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "watchsimulator",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "watchsimulator*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_watchos.test_runner_support,
            "UIDeviceFamily:0": "4",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_base_bundle_id_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ui_test_with_base_bundle_id_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.ui-test-with-base-bundle-id-derived-bundle-id",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
