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

"""ios_app_clip Starlark tests."""

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
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)
load(
    "//test/starlark_tests/rules:linkmap_test.bzl",
    "linkmap_test",
)
load(
    ":common.bzl",
    "common",
)

def ios_app_clip_test_suite(name):
    """Test suite for ios_app_clip.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Tests that app clip is codesigned when built as a standalone app
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        verifier_script = "verifier_scripts/app_clip_codesign_verifier.sh",
        tags = [name],
    )

    # Tests that app clip entitlements are added when built for simulator.
    apple_verification_test(
        name = "{}_app_clip_entitlements_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        verifier_script = "verifier_scripts/app_clip_entitlements_verifier.sh",
        tags = [name],
    )

    # Tests that app clip entitlements are added when built for device.
    apple_verification_test(
        name = "{}_app_clip_entitlements_simulator_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        verifier_script = "verifier_scripts/app_clip_entitlements_verifier.sh",
        tags = [name],
    )

    # Tests that entitlements are present when specified and built for simulator.
    apple_verification_test(
        name = "{}_entitlements_simulator_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    # Tests that entitlements are present when specified and built for device.
    apple_verification_test(
        name = "{}_entitlements_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    # Tests that the linkmap outputs are produced when `--objc_generate_linkmap`
    # is present.
    linkmap_test(
        name = "{}_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_clip_x86_64.linkmap",
            "app_clip_arm64.linkmap",
        ],
        tags = [name],
    )

    # Verifies that Info.plist contains correct package type
    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "app_clip",
            "CFBundleIdentifier": "com.google.example.clip",
            "CFBundleName": "app_clip",
            "CFBundlePackageType": "APPL",
            "CFBundleSupportedPlatforms:0": "iPhone*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "iphone*",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "iphone*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_ios.appclip_support,
            "UIDeviceFamily:0": "1",
        },
        tags = [name],
    )

    # Tests that the provisioning profile is present when built for device.
    archive_contents_test(
        name = "{}_contains_provisioning_profile_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip",
        contains = [
            "$BUNDLE_ROOT/embedded.mobileprovision",
        ],
        tags = [name],
    )

    # Tests that the provisioning profile is present when built for device and embedded in an app.
    archive_contents_test(
        name = "{}_embedding_provisioning_profile_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_clip",
        contains = [
            "$BUNDLE_ROOT/AppClips/app_clip.app/embedded.mobileprovision",
        ],
        tags = [name],
    )

    # Verify that Swift dylibs are packaged with the application, when the application uses Swift.
    archive_contents_test(
        name = "{}_device_swift_dylibs_present".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_clip_with_swift_support",
        contains = [
            "$ARCHIVE_ROOT/SwiftSupport/iphoneos/libswift_Concurrency.dylib",
            "$BUNDLE_ROOT/AppClips/app_clip_with_swift_support.app/Frameworks/libswift_Concurrency.dylib",
            "$BUNDLE_ROOT/Frameworks/libswift_Concurrency.dylib",
        ],
        not_contains = [
            "$BUNDLE_ROOT/AppClips/SwiftSupport/iphoneos/libswift_Concurrency.dylib",
        ],
        tags = [name],
    )

    # Test dSYM binaries and linkmaps from framework embedded via 'data' are propagated correctly
    # at the top-level ios_extension rule, and present through the 'dsysms' and 'linkmaps' output
    # groups.
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_dsyms_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "dsyms",
        expected_outputs = [
            "app_clip_with_fmwks_from_objc_swift_libraries_using_data.app.dSYM/Contents/Info.plist",
            "app_clip_with_fmwks_from_objc_swift_libraries_using_data.app.dSYM/Contents/Resources/DWARF/app_clip_with_fmwks_from_objc_swift_libraries_using_data",
            "fmwk_min_os_baseline_with_bundle.framework.dSYM/Contents/Info.plist",
            "fmwk_min_os_baseline_with_bundle.framework.dSYM/Contents/Resources/DWARF/fmwk_min_os_baseline_with_bundle",
            "fmwk_no_version.framework.dSYM/Contents/Info.plist",
            "fmwk_no_version.framework.dSYM/Contents/Resources/DWARF/fmwk_no_version",
            "fmwk_with_resources.framework.dSYM/Contents/Info.plist",
            "fmwk_with_resources.framework.dSYM/Contents/Resources/DWARF/fmwk_with_resources",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_clip_with_fmwks_from_objc_swift_libraries_using_data_arm64.linkmap",
            "app_clip_with_fmwks_from_objc_swift_libraries_using_data_x86_64.linkmap",
            "fmwk_min_os_baseline_with_bundle_arm64.linkmap",
            "fmwk_min_os_baseline_with_bundle_x86_64.linkmap",
            "fmwk_no_version_arm64.linkmap",
            "fmwk_no_version_x86_64.linkmap",
            "fmwk_with_resources_arm64.linkmap",
            "fmwk_with_resources_x86_64.linkmap",
        ],
        tags = [name],
    )

    # Test transitive frameworks dSYM bundles are propagated by the AppleDsymBundleInfo provider.
    apple_dsym_bundle_info_test(
        name = "{}_with_runtime_framework_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip_with_fmwks_from_objc_swift_libraries_using_data",
        expected_direct_dsyms = [
            "dSYMs/app_clip_with_fmwks_from_objc_swift_libraries_using_data.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app_clip_with_fmwks_from_objc_swift_libraries_using_data.app.dSYM",
            "dSYMs/fmwk_min_os_baseline_with_bundle.framework.dSYM",
            "dSYMs/fmwk_no_version.framework.dSYM",
            "dSYMs/fmwk_with_resources.framework.dSYM",
        ],
        tags = [name],
    )

    # Tests inclusion of extensions.
    archive_contents_test(
        name = "{}_contains_ios_extension".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_clip_with_ext",
        contains = [
            "$BUNDLE_ROOT/PlugIns/swift_ios_app_clip_ext.appex/swift_ios_app_clip_ext",
        ],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
