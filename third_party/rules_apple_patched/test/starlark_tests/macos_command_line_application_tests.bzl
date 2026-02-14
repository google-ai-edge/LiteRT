# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""macos_command_line_application Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:analysis_runfiles_test.bzl",
    "analysis_runfiles_dsym_test",
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
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "binary_contents_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def macos_command_line_application_test_suite(name):
    """Test suite for macos_command_line_application.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_swift_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic_swift",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    binary_contents_test(
        name = "{}_merged_info_plist_binary_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_info_plists",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        embedded_plist_test_values = {
            "AnotherKey": "AnotherValue",
            "BuildMachineOSBuild": "*",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "cmd_app_info_plists",
            "CFBundleVersion": "1.0",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "macosx",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "macosx*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "LSMinimumSystemVersion": common.min_os_macos.baseline,
        },
        tags = [name],
    )

    binary_contents_test(
        name = "{}_merged_info_and_launchd_plists_info_plist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_info_and_launchd_plists",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        embedded_plist_test_values = {
            "AnotherKey": "AnotherValue",
            "BuildMachineOSBuild": "*",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "cmd_app_info_and_launchd_plists",
            "CFBundleVersion": "1.0",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "macosx",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "macosx*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "LSMinimumSystemVersion": "10.13",
        },
        tags = [name],
    )

    binary_contents_test(
        name = "{}_merged_info_and_launchd_plists_launchd_plist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_info_and_launchd_plists",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        embedded_plist_test_values = {
            "AnotherKey": "AnotherValue",
            "Label": "com.test.bundle",
        },
        plist_section_name = "__launchd_plist",
        tags = [name],
    )

    binary_contents_test(
        name = "{}_custom_linkopts_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_special_linkopts",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_linkopts_test_main"],
        tags = [name],
    )

    binary_contents_test(
        name = "{}_exported_symbols_list_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_dead_stripped",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionMain"],
        binary_not_contains_symbols = ["_dontCallMeMain"],
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic",
        output_group_name = "dsyms",
        expected_outputs = [
            "cmd_app_basic.dSYM/Contents/Info.plist",
            "cmd_app_basic.dSYM/Contents/Resources/DWARF/cmd_app_basic",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic",
        expected_direct_dsyms = ["dSYMs/cmd_app_basic.dSYM"],
        expected_transitive_dsyms = ["dSYMs/cmd_app_basic.dSYM"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_infoplist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_info_plists",
        expected_values = {
            "AnotherKey": "AnotherValue",
            "BuildMachineOSBuild": "*",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "cmd_app_info_plists",
            "CFBundleShortVersionString": "1.0",
            "CFBundleSupportedPlatforms:0": "MacOSX",
            "CFBundleVersion": "1.0",
            "DTPlatformVersion": "*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "DTPlatformName": "macosx",
            "DTSDKBuild": "*",
            "DTSDKName": "macosx*",
            "LSMinimumSystemVersion": "10.13",
        },
        tags = [name],
    )

    analysis_runfiles_dsym_test(
        name = "{}_runfiles_dsym_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic",
        expected_runfiles = [
            "test/starlark_tests/targets_under_test/macos/cmd_app_basic.dSYM/Contents/Resources/DWARF/cmd_app_basic",
            "test/starlark_tests/targets_under_test/macos/cmd_app_basic.dSYM/Contents/Info.plist",
        ],
        tags = [name],
    )

    binary_contents_test(
        name = "{}_version_plist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic_version",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        embedded_plist_test_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleShortVersionString": "1.2",
            "CFBundleVersion": "1.2.3",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "macosx",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "macosx*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "LSMinimumSystemVersion": "10.13",
        },
        tags = [name],
    )

    binary_contents_test(
        name = "{}_base_bundle_id_derived_bundle_id_plist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_with_base_bundle_id_derived_bundle_id",
        binary_test_file = "$BINARY",
        compilation_mode = "opt",
        embedded_plist_test_values = {
            "CFBundleIdentifier": "com.bazel.app.example.cmd-app-with-base-bundle-id-derived-bundle-id",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_default_dsym_version_in_info_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic",
        apple_generate_dsym = True,
        output_group_name = "dsyms",
        plist_test_file_shortpath = "test/starlark_tests/targets_under_test/macos/cmd_app_basic.dSYM/Contents/Info.plist",
        expected_values = {
            "CFBundleIdentifier": "com.apple.xcode.dsym.cmd_app_basic.dSYM",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_dsym_version_in_info_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:cmd_app_basic_version",
        apple_generate_dsym = True,
        output_group_name = "dsyms",
        plist_test_file_shortpath = "test/starlark_tests/targets_under_test/macos/cmd_app_basic_version.dSYM/Contents/Info.plist",
        expected_values = {
            "CFBundleIdentifier": "com.apple.xcode.dsym.cmd_app_basic_version.dSYM",
            "CFBundleShortVersionString": "1.2",
            "CFBundleVersion": "1.2.3",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
