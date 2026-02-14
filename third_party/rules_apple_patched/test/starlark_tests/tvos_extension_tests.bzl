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

"""tvos_extension Starlark tests."""

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
    "entry_point_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def tvos_extension_test_suite(name):
    """Test suite for tvos_extension.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        output_group_name = "dsyms",
        expected_outputs = [
            "ext.appex.dSYM/Contents/Info.plist",
            "ext.appex.dSYM/Contents/Resources/DWARF/ext",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        expected_direct_dsyms = ["dSYMs/ext.appex.dSYM"],
        expected_transitive_dsyms = ["dSYMs/ext.appex.dSYM"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "ext",
            "CFBundleIdentifier": "com.google.example.ext",
            "CFBundleName": "ext",
            "CFBundlePackageType": "XPC!",
            "CFBundleSupportedPlatforms:0": "AppleTVSimulator*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "appletvsimulator*",
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

    # Tests that the provisioning profile is present when built for device.
    archive_contents_test(
        name = "{}_contains_provisioning_profile_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        contains = [
            "$BUNDLE_ROOT/embedded.mobileprovision",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_correct_rpath_header_value_test".format(name),
        build_type = "device",
        binary_test_file = "$CONTENT_ROOT/ext",
        macho_load_commands_contain = [
            "path @executable_path/Frameworks (offset 12)",
            "path @executable_path/../../Frameworks (offset 12)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        tags = [name],
    )

    # Verify that Swift dylibs are packaged with the application, not with the extension, when only
    # an extension uses Swift. And to be safe, verify that they aren't packaged with the extension.
    archive_contents_test(
        name = "{}_device_swift_dylibs_present".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_swift_ext",
        not_contains = ["$BUNDLE_ROOT/PlugIns/ext.appex/Frameworks/libswiftCore.dylib"],
        contains = [
            "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
            "$ARCHIVE_ROOT/SwiftSupport/appletvos/libswiftCore.dylib",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_simulator_swift_dylibs_present".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_swift_ext",
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        not_contains = ["$BUNDLE_ROOT/PlugIns/ext.appex/Frameworks/libswiftCore.dylib"],
        tags = [name],
    )

    # Test dSYM binaries and linkmaps from framework embedded via 'data' are propagated correctly
    # at the top-level tvos_extension rule, and present through the 'dsysms' and 'linkmaps' output
    # groups.
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_dsyms_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "dsyms",
        expected_outputs = [
            "ext_with_fmwks_from_objc_swift_libraries_using_data.appex.dSYM/Contents/Info.plist",
            "ext_with_fmwks_from_objc_swift_libraries_using_data.appex.dSYM/Contents/Resources/DWARF/ext_with_fmwks_from_objc_swift_libraries_using_data",
            "fmwk_with_resource_bundles.framework.dSYM/Contents/Info.plist",
            "fmwk_with_resource_bundles.framework.dSYM/Contents/Resources/DWARF/fmwk_with_resource_bundles",
            "fmwk_with_structured_resources.framework.dSYM/Contents/Info.plist",
            "fmwk_with_structured_resources.framework.dSYM/Contents/Resources/DWARF/fmwk_with_structured_resources",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "linkmaps",
        expected_outputs = [
            "ext_with_fmwks_from_objc_swift_libraries_using_data_arm64.linkmap",
            "ext_with_fmwks_from_objc_swift_libraries_using_data_x86_64.linkmap",
            "fmwk_with_resource_bundles_arm64.linkmap",
            "fmwk_with_resource_bundles_x86_64.linkmap",
            "fmwk_with_structured_resources_arm64.linkmap",
            "fmwk_with_structured_resources_x86_64.linkmap",
        ],
        tags = [name],
    )

    # Test transitive frameworks dSYM bundles are propagated by the AppleDsymBundleInfo provider.
    apple_dsym_bundle_info_test(
        name = "{}_with_runtime_framework_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext_with_fmwks_from_objc_swift_libraries_using_data",
        expected_direct_dsyms = [
            "dSYMs/ext_with_fmwks_from_objc_swift_libraries_using_data.appex.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/ext_with_fmwks_from_objc_swift_libraries_using_data.appex.dSYM",
            "dSYMs/fmwk_with_resource_bundles.framework.dSYM",
            "dSYMs/fmwk_with_structured_resources.framework.dSYM",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.ext-with-capability-set-derived-bundle-id",
        },
        tags = [name],
    )

    # Test that an ExtensionKit extension is bundled in Extensions and not PlugIns.
    archive_contents_test(
        name = "{}_extensionkit_bundling_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_extensionkit_ext",
        contains = ["$BUNDLE_ROOT/Extensions/extensionkit_ext.appex/extensionkit_ext"],
        not_contains = ["$BUNDLE_ROOT/PlugIns/extensionkit_ext.appex/extensionkit_ext"],
        tags = [name],
    )

    # Test that the default entry point is _NSExtensionMain (modern behavior).
    entry_point_test(
        name = "{}_default_entry_point_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext",
        entry_point = "_NSExtensionMain",
        tags = [name],
    )

    # Test that the entry point is _TVExtensionMain when legacy_entry_point=True.
    entry_point_test(
        name = "{}_legacy_entry_point_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:ext_with_legacy_entry_point",
        entry_point = "_TVExtensionMain",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
