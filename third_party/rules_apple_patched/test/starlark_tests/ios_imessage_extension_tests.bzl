# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""ios_imessage_extension Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_actions_test.bzl",
    "analysis_target_actions_test",
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
load(
    ":common.bzl",
    "common",
)

visibility("private")

def ios_imessage_extension_test_suite(name):
    """Test suite for ios_imessage_extension.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:imessage_ext",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    # Tests xcasset tool is passed the correct arguments.
    analysis_target_actions_test(
        name = "{}_xcasset_actool_argv".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:imessage_ext",
        target_mnemonic = "AssetCatalogCompile",
        expected_argv = [
            "xctoolrunner actool --compile",
            "--include-sticker-content",
            "--stickers-icon-role",
            "extension",
            "--minimum-deployment-target " + common.min_os_ios.baseline,
            "--platform iphonesimulator",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:imessage_ext",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "imessage-ext",
            "CFBundleIdentifier": "com.google.example.MessagesExtension",
            "CFBundleName": "imessage-ext",
            "CFBundlePackageType": "XPC!",
            "CFBundleSupportedPlatforms:0": "iPhone*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "iphone*",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "iphone*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_ios.baseline,
            "NSExtension:NSExtensionPointIdentifier": "com.apple.widget-extension",
            "UIDeviceFamily:0": "1",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:imessage_ext_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.MessagesExtension",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_imessage_icons_in_ipa_test".format(name),
        build_type = "device",
        contains = [
            "$BUNDLE_ROOT/app_icon27x20@2x.png",
            "$BUNDLE_ROOT/app_icon32x24@2x.png",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:imessage_ext",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_imessage_wrong_icons_in_ipa_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:imessage_ext_with_wrong_appicons",
        expected_error = """
Found in app_icons a file that cannot be used as an app icon:
test/testdata/resources/app_icons_ios.xcassets/app_icon.appiconset/Contents.json

Valid icon bundles for this target have the following extensions: [".stickersiconset/"]
""",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
