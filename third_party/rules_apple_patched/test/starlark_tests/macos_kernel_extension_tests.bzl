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

"""macos_kernel_extension Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_target_actions_test.bzl",
    "analysis_target_actions_test",
    "make_analysis_target_actions_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_outputs_test.bzl",
    "analysis_target_outputs_test",
)
load(
    "//test/starlark_tests/rules:apple_verification_test.bzl",
    "apple_verification_test",
)
load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)

_analysis_arm64_macos_cpu_test = make_analysis_target_actions_test(
    config_settings = {"//command_line_option:macos_cpus": "arm64"},
)
_analysis_x86_64_macos_cpu_test = make_analysis_target_actions_test(
    config_settings = {"//command_line_option:macos_cpus": "x86_64"},
)

def macos_kernel_extension_test_suite(name):
    """Test suite for macos_kernel_extension.

    Args:
      name: the base name to be used in things created by this macro
    """
    analysis_target_outputs_test(
        name = "{}_zip_file_output_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext",
        expected_outputs = ["kext.zip"],
        tags = [name],
    )

    _analysis_arm64_macos_cpu_test(
        name = "{}_arm64_macos_cpu_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext",
        target_mnemonic = "ObjcLink",
        expected_argv = [
            "/wrapped_clang",
            " -kext",
            " -target arm64e-apple-macosx10.13",
        ],
        tags = [name],
    )

    _analysis_x86_64_macos_cpu_test(
        name = "{}_x86_64_macos_cpu_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext",
        target_mnemonic = "ObjcLink",
        expected_argv = [
            "/wrapped_clang",
            " -kext",
            " -target x86_64-apple-macosx10.13",
        ],
        tags = [name],
    )

    analysis_target_actions_test(
        name = "{}_default_macos_cpu_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext",
        target_mnemonic = "ObjcLink",
        expected_argv = [
            "/wrapped_clang",
            " -kext",
            " -target x86_64-apple-macosx10.13",
        ],
        tags = [name],
    )

    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_plist_test".format(name),
        build_type = "device",
        plist_test_file = "$CONTENT_ROOT/Info.plist",
        plist_test_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "kext",
            "CFBundleIdentifier": "com.google.kext",
            "CFBundleName": "kext",
            "CFBundlePackageType": "KEXT",
            "CFBundleSupportedPlatforms:0": "MacOSX",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "macosx",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "macosx*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "LSMinimumSystemVersion": "10.13",
            "IOKitPersonalities": "*",
            "OSBundleLibraries": "*",
        },
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_exported_symbols_list_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext_dead_stripped",
        binary_test_file = "$CONTENT_ROOT/MacOS/kext_dead_stripped",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        binary_not_contains_symbols = ["_dontCallMeShared"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:kext_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
