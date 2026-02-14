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

"""ios_application Starlark tests."""

load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)
load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_actions_test.bzl",
    "analysis_target_actions_tree_artifacts_outputs_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_outputs_test.bzl",
    "analysis_target_outputs_test",
    "analysis_target_tree_artifacts_outputs_test",
)
load(
    "//test/starlark_tests/rules:apple_codesigning_dossier_info_provider_test.bzl",
    "apple_codesigning_dossier_info_provider_test",
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
    "apple_symbols_file_test",
    "archive_contents_test",
)
load(
    "//test/starlark_tests/rules:entitlements_contents_test.bzl",
    "entitlements_contents_test",
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
    "//test/starlark_tests/rules:output_group_zip_contents_test.bzl",
    "output_group_zip_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def ios_application_test_suite(name):
    """Test suite for ios_application.

    Args:
      name: the base name to be used in things created by this macro
    """
    analysis_target_outputs_test(
        name = "{}_ipa_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        expected_outputs = ["app.ipa"],
        tags = [name],
    )
    analysis_target_tree_artifacts_outputs_test(
        name = "{}_tree_artifact_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        expected_outputs = ["app_minimal.app"],
        tags = [name],
    )
    analysis_target_tree_artifacts_outputs_test(
        name = "{}_tree_artifact_outputs_with_space_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal_with_space_and_post_processor",
        expected_outputs = ["app_minimal_with_space_and_post_processor_archive-root/Payload/app minimal.app"],
        tags = [name],
    )

    analysis_target_tree_artifacts_outputs_test(
        name = "{}_duplicate_bundle_target1_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_duplicate_bundle",
        expected_outputs = ["app_with_duplicate_bundle.app"],
        tags = [name],
    )

    analysis_target_tree_artifacts_outputs_test(
        name = "{}_duplicate_bundle_target2_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_duplicate_bundle_beta",
        expected_outputs = ["app_with_duplicate_bundle_beta_archive-root/Payload/app_with_duplicate_bundle.app"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_tree_artifact_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        build_settings = {
            build_settings_labels.use_tree_artifacts_outputs: "True",
        },
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/PkgInfo",
            "$BUNDLE_ROOT/app_minimal",
        ],
        tags = [name],
    )

    analysis_target_actions_tree_artifacts_outputs_test(
        name = "{}_registers_action_for_tree_artifact_bundling_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        target_mnemonic = "BundleTreeApp",
        not_expected_mnemonic = ["BundleApp"],
        tags = [name],
    )

    apple_verification_test(
        name = "{}_ipa_opt_strip_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        compilation_mode = "opt",
        objc_enable_binary_stripping = True,
        tags = [name],
    )

    # Tests that an app with a mixed target framework compiles
    analysis_target_outputs_test(
        name = "{}_mixed_target_framework_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_with_multiple_objc_library_and_swift_library_deps",
        expected_outputs = ["app_with_fmwk_with_multiple_objc_library_and_swift_library_deps.ipa"],
        tags = [name],
    )

    apple_verification_test(
        name = "{}_ext_and_fmwk_provisioned_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_and_fmwk_provisioned",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_ext_and_fmwk_provisioned_codesign_asan_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_and_fmwk_provisioned",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        sanitizer = "asan",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_codesignopts_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_codesignopts",
        verifier_script = "verifier_scripts/codesignopts_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_imported_fmwk_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    # Verify ios_application with imported dynamic framework bundles files for Objective-C/Swift
    archive_contents_test(
        name = "{}_with_imported_dynamic_fmwk_bundles_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_fmwk",
        contains = [
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/iOSDynamicFramework",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Resources/iOSDynamicFramework.bundle/Info.plist",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Headers/SharedClass.h",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Headers/iOSDynamicFramework.h",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Modules/module.modulemap",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_with_imported_dynamic_fmwk_bundles_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_dynamic_fmwk",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/iOSDynamicFramework",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Headers/SharedClass.h",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Headers/iOSDynamicFramework.h",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/Modules/module.modulemap",
        ],
        tags = [name],
    )

    # Verify ios_application with imported static framework contains symbols for Objective-C/Swift,
    # and resource bundles; but does not bundle the static library.
    archive_contents_test(
        name = "{}_with_imported_static_fmwk_contains_symbols_and_bundles_resources".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_static_fmwk",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        is_not_binary_plist = ["$BUNDLE_ROOT/iOSStaticFramework.bundle/Info.plist"],
        contains = ["$BUNDLE_ROOT/iOSStaticFramework.bundle/Info.plist"],
        not_contains = ["$BUNDLE_ROOT/Frameworks/iOSStaticFramework.framework"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_with_imported_swift_static_fmwk_contains_symbols_and_not_bundles_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_swift_static_fmwk",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_OBJC_CLASS_$__TtC23iOSSwiftStaticFramework11SharedClass",
        ],
        not_contains = ["$BUNDLE_ROOT/Frameworks/iOSSwiftStaticFramework.framework"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_with_imported_swift_static_fmwk_and_no_swift_module_interface_file_contains_symbols_and_not_bundles_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_swift_static_fmwk_without_module_interface_files",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_OBJC_CLASS_$__TtC23iOSSwiftStaticFramework11SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        not_contains = ["$BUNDLE_ROOT/Frameworks/iOSSwiftStaticFramework.framework"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_with_imported_static_fmwk_contains_symbols_and_not_bundles_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_static_fmwk",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        not_contains = ["$BUNDLE_ROOT/Frameworks/iOSStaticFramework.framework"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_with_imported_swift_static_fmwk_contains_symbols_and_not_bundles_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_swift_static_fmwk",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_OBJC_CLASS_$__TtC23iOSSwiftStaticFramework11SharedClass"],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        not_contains = ["$BUNDLE_ROOT/Frameworks/iOSSwiftStaticFramework.framework"],
        tags = [name],
    )

    # Verify ios_application with imported static framework that has data attribute
    # bundles the framework's own .bundle/ and its data resources in the final binary.
    archive_contents_test(
        name = "{}_swift_with_imported_static_fmwk_with_bundle_and_data".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_static_fmwk_with_data",
        contains = [
            "$BUNDLE_ROOT/sample.png",
            "$BUNDLE_ROOT/it.lproj/view_ios.nib",
            "$BUNDLE_ROOT/fr.lproj/view_ios.nib",
            "$BUNDLE_ROOT/iOSStaticFramework.bundle/",
        ],
        tags = [name],
    )

    apple_verification_test(
        name = "{}_fmwk_with_imported_fmwk_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_importing_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_fmwk_in_fmwk_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_with_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_fmwk_in_fmwk_provisioned_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_with_fmwk_provisioned",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_two_fmwk_provisioned_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_two_fmwk_provisioned",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    # Tests that Swift standard libraries bundled in SwiftSupport have the code
    # signature from Apple.
    archive_contents_test(
        name = "{}_swift_support_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_swift_dep",
        binary_test_file = "$ARCHIVE_ROOT/SwiftSupport/iphoneos/libswiftCore.dylib",
        codesign_info_contains = [
            "Identifier=com.apple.dt.runtime.swiftCore",
            "Authority=Software Signing",
            "Authority=Apple Code Signing Certification Authority",
            "Authority=Apple Root CA",
            "TeamIdentifier=59GAB85EFG",
        ],
        tags = [name],
    )

    apple_verification_test(
        name = "{}_entitlements_simulator_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_entitlements_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    entitlements_contents_test(
        name = "{}_entitlements_contents_simulator_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        expected_values = {
            "application-identifier": "FOOBARBAZ1.*",
            "get-task-allow": "true",
            "test-an-entitlement": "false",
        },
        tags = [name],
    )

    entitlements_contents_test(
        name = "{}_entitlements_contents_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        expected_values = {
            "application-identifier": "FOOBARBAZ1.*",
            "get-task-allow": "true",
            "test-an-entitlement": "false",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_custom_executable_name_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_custom_executable_name",
        contains = ["$BUNDLE_ROOT/app.exe"],
        not_contains = ["$BUNDLE_ROOT/app_with_custom_executable_name"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_dbg_resources_simulator_test".format(name),
        build_type = "simulator",
        compilation_mode = "dbg",
        is_binary_plist = ["$BUNDLE_ROOT/resource_bundle.bundle/Info.plist"],
        is_not_binary_plist = ["$BUNDLE_ROOT/Another.plist"],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_opt_resources_simulator_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        is_binary_plist = [
            "$BUNDLE_ROOT/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_dbg_resources_device_test".format(name),
        build_type = "device",
        compilation_mode = "dbg",
        is_binary_plist = ["$BUNDLE_ROOT/resource_bundle.bundle/Info.plist"],
        is_not_binary_plist = ["$BUNDLE_ROOT/Another.plist"],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_opt_resources_device_test".format(name),
        build_type = "device",
        compilation_mode = "opt",
        is_binary_plist = [
            "$BUNDLE_ROOT/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_opt_strings_simulator_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        contains = [
            "$RESOURCE_ROOT/localization.bundle/en.lproj/files.stringsdict",
            "$RESOURCE_ROOT/localization.bundle/en.lproj/greetings.strings",
        ],
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        output_group_name = "dsyms",
        expected_outputs = [
            "app.app.dSYM/Contents/Info.plist",
            "app.app.dSYM/Contents/Resources/DWARF/app",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        expected_direct_dsyms = [
            "dSYMs/app.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app.app.dSYM",
        ],
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_transitive_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_and_fmwk_provisioned",
        output_group_name = "dsyms",
        expected_outputs = [
            "app_with_ext_and_fmwk_provisioned.app.dSYM/Contents/Info.plist",
            "app_with_ext_and_fmwk_provisioned.app.dSYM/Contents/Resources/DWARF/app_with_ext_and_fmwk_provisioned",
            "ext_with_fmwk_provisioned.appex.dSYM/Contents/Info.plist",
            "ext_with_fmwk_provisioned.appex.dSYM/Contents/Resources/DWARF/ext_with_fmwk_provisioned",
            "fmwk_with_provisioning.framework.dSYM/Contents/Info.plist",
            "fmwk_with_provisioning.framework.dSYM/Contents/Resources/DWARF/fmwk_with_provisioning",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_transitive_dsyms_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_and_fmwk_provisioned",
        expected_direct_dsyms = [
            "dSYMs/app_with_ext_and_fmwk_provisioned.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/fmwk_with_provisioning.framework.dSYM",
            "dSYMs/ext_with_fmwk_provisioned.appex.dSYM",
            "dSYMs/app_with_ext_and_fmwk_provisioned.app.dSYM",
        ],
        tags = [name],
    )

    apple_dsym_bundle_info_test(
        name = "{}_custom_executable_name_dsyms_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_custom_executable_name",
        expected_direct_dsyms = [
            "dSYMs/app_with_custom_executable_name_dsyms/custom_bundle_name.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app_with_custom_executable_name_dsyms/custom_bundle_name.app.dSYM",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_custom_executable_name_dsyms_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_custom_executable_name",
        output_group_name = "dsyms",
        expected_outputs = [
            "app_with_custom_executable_name_dsyms/custom_bundle_name.app.dSYM/Contents/Resources/DWARF/app.exe",
            "app_with_custom_executable_name_dsyms/custom_bundle_name.app.dSYM/Contents/Info.plist",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "app",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "app",
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
            "MinimumOSVersion": common.min_os_ios.baseline,
            "UIDeviceFamily:0": "1",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_custom_executable_name_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_custom_executable_name",
        expected_values = {
            "CFBundleExecutable": "app.exe",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_multiple_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_multiple_infoplists",
        expected_values = {
            "AnotherKey": "AnotherValue",
            "CFBundleExecutable": "app_multiple_infoplists",
        },
        tags = [name],
    )

    # Tests that the archive contains .symbols package files when `include_symbols_in_bundle`
    # is enabled for both the iOS application and the watchOS extensions.
    apple_symbols_file_test(
        name = "{}_archive_contains_apple_symbols_files_watchos_test".format(name),
        binary_paths = [
            "Payload/companion.app/companion",
            "Payload/companion.app/Watch/app.app/PlugIns/ext.appex/ext",
            "Payload/companion.app/Watch/app.app/PlugIns/ext.appex/PlugIns/watchos_app_extension.appex/watchos_app_extension",
        ],
        build_type = "simulator",
        tags = [name],
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ios_watchos_with_watchos_extension_and_symbols_in_bundle",
    )

    # Tests that the archive contains .symbols package files when `include_symbols_in_bundle`
    # is enabled.
    apple_symbols_file_test(
        name = "{}_archive_contains_apple_symbols_files_test".format(name),
        binary_paths = [
            "Payload/app_with_ext_and_fmwk_and_symbols_in_bundle.app/app_with_ext_and_fmwk_and_symbols_in_bundle",
            "Payload/app_with_ext_and_fmwk_and_symbols_in_bundle.app/PlugIns/ext_with_fmwk_provisioned.appex/ext_with_fmwk_provisioned",
            "Payload/app_with_ext_and_fmwk_and_symbols_in_bundle.app/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
        ],
        build_type = "simulator",
        tags = [name],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_and_fmwk_and_symbols_in_bundle",
    )

    # Tests that the archive contains .symbols package files generated from
    # imported frameworks when `include_symbols_in_bundle` is enabled.
    apple_symbols_file_test(
        name = "{}_archive_contains_apple_symbols_files_from_external_fmwk_test".format(name),
        binary_paths = [
            "Payload/app_with_imported_dynamic_fmwk_with_dsym.app/app_with_imported_dynamic_fmwk_with_dsym",
            "Payload/app_with_imported_dynamic_fmwk_with_dsym.app/Frameworks/iOSDynamicFramework.framework/iOSDynamicFramework",
            "Payload/app_with_imported_dynamic_fmwk_with_dsym.app/Frameworks/iOSDynamicFrameworkWithDebugInfo.framework/iOSDynamicFrameworkWithDebugInfo",
        ],
        build_type = "simulator",
        tags = [name],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_dynamic_fmwk_with_dsym",
    )

    # Tests that the linkmap outputs are produced when `--objc_generate_linkmap`
    # is present.
    linkmap_test(
        name = "{}_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_x86_64.linkmap",
            "app_arm64.linkmap",
        ],
        tags = [name],
    )

    # Test that Bitcode was removed from the imported framework when building
    # with Bitcode disabled.
    archive_contents_test(
        name = "{}_imported_dynamic_framework_bitcode_strip_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_dynamic_fmwk_with_bitcode",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/iOSDynamicFrameworkWithBitcode.framework/iOSDynamicFrameworkWithBitcode",
        macho_load_commands_not_contain = ["segname __LLVM"],
        tags = [name],
    )

    # Test that Bitcode was removed from the Swift standard libraries when building
    # with Bitcode disabled.
    archive_contents_test(
        name = "{}_swift_stdlibs_bitcode_strip_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_swift_dep",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
        macho_load_commands_not_contain = ["segname __LLVM"],
        tags = [name],
    )

    # Tests that the provisioning profile is present when built for device.
    archive_contents_test(
        name = "{}_contains_provisioning_profile_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        contains = [
            "$BUNDLE_ROOT/embedded.mobileprovision",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_contains_asan_dylib_simulator_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.asan_iossim_dynamic.dylib",
        ],
        sanitizer = "asan",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_contains_asan_dylib_device_test".format(name),
        build_type = "device",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.asan_ios_dynamic.dylib",
        ],
        sanitizer = "asan",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_contains_tsan_dylib_device_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.tsan_iossim_dynamic.dylib",
        ],
        sanitizer = "tsan",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_contains_ubsan_dylib_simulator_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.ubsan_iossim_dynamic.dylib",
        ],
        sanitizer = "ubsan",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_contains_ubsan_dylib_device_test".format(name),
        build_type = "device",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.ubsan_ios_dynamic.dylib",
        ],
        sanitizer = "ubsan",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal",
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_with_minimum_deployment_os_version".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal_with_deployment_version",
        tags = [name],
        expected_values = {
            "MinimumOSVersion": "14.0",
        },
    )

    # Tests analysis phase failure when an extension depends on a framework which
    # is not marked extension_safe.
    analysis_failure_message_test(
        name = "{}_fails_with_extension_depending_on_not_extension_safe_framework".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_with_fmwk_not_extension_safe",
        expected_error = (
            "The target {target} is for an extension but its framework dependency " +
            "{framework} is not marked extension-safe. Specify 'extension_safe " +
            "= True' on the framework target."
        ).format(
            framework = Label("//test/starlark_tests/targets_under_test/ios:fmwk_not_extension_safe"),
            target = Label("//test/starlark_tests/targets_under_test/ios:ext_with_fmwk_not_extension_safe"),
        ),
        tags = [name],
    )

    output_group_zip_contents_test(
        name = "{}_has_combined_zip_output_group".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        output_group_name = "combined_dossier_zip",
        output_group_file_shortpath = "test/starlark_tests/targets_under_test/ios/app_dossier_with_bundle.zip",
        contains = [
            "bundle/Payload/app.app/Info.plist",
            "bundle/Payload/app.app/app",
            "dossier/manifest.json",
        ],
        tags = [name],
    )

    # Test app with App Intents generates and bundles Metadata.appintents bundle.
    archive_contents_test(
        name = "{}_contains_app_intents_metadata_bundle_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_intents",
        contains = [
            "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
            "$BUNDLE_ROOT/Metadata.appintents/version.json",
        ],
        tags = [name],
    )

    # Test app with a Widget Configuration Intent with a computed property generates and bundles Metadata.appintents bundle.
    archive_contents_test(
        name = "{}_with_widget_configuration_intent_contains_app_intents_metadata_bundle_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_widget_configuration_intent",
        contains = [
            "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
            "$BUNDLE_ROOT/Metadata.appintents/version.json",
        ],
        tags = [
            name,
        ],
    )

    # Test app that has two Intents defined as top level modules generates an error message.
    analysis_failure_message_test(
        name = "{}_with_two_app_intents_and_two_modules_fails".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_intent_and_widget_configuration_intent",
        expected_error = (
            "App Intents must have only one module name for metadata generation to work correctly."
        ).format(
            package = "//test/starlark_tests/targets_under_test/ios",
        ),
        tags = [
            name,
        ],
    )

    # Test app with App Intents generates and bundles Metadata.appintents bundle for fat binaries.
    archive_contents_test(
        name = "{}_fat_build_contains_app_intents_metadata_bundle_test".format(name),
        build_type = "simulator",
        cpus = {
            "ios_multi_cpus": ["x86_64", "sim_arm64"],
        },
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_intents",
        contains = [
            "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
            "$BUNDLE_ROOT/Metadata.appintents/version.json",
        ],
        tags = [name],
    )

    # Test Metadata.appintents bundle contents for simulator and device.
    archive_contents_test(
        name = "{}_metadata_appintents_bundle_contents_for_simulator_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_intents",
        text_test_file = "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
        text_test_values = [
            ".*HelloWorldIntent.*",
            ".*IntelIntent.*",
            ".*iOSIntent.*",
        ],
        text_file_not_contains = [
            ".*ArmIntent.*",
            ".*macOSIntent.*",
            ".*tvOSIntent.*",
            ".*watchOSIntent.*",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_metadata_appintents_bundle_contents_for_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_app_intents",
        text_test_file = "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
        text_test_values = [
            ".*HelloWorldIntent.*",
            ".*ArmIntent.*",
            ".*iOSIntent.*",
        ],
        text_file_not_contains = [
            ".*IntelIntent.*",
            ".*macOSIntent.*",
            ".*tvOSIntent.*",
            ".*watchOSIntent.*",
        ],
        tags = [name],
    )

    # Test dSYM binaries and linkmaps from framework embedded via 'data' are propagated correctly
    # at the top-level ios_application rule, and present through the 'dsysms' and 'linkmaps' output
    # groups.
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_dsyms_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        output_group_name = "dsyms",
        expected_outputs = [
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM/Contents/Info.plist",
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM/Contents/Resources/DWARF/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
            # Frameworks
            "fmwk.framework.dSYM/Contents/Info.plist",
            "fmwk.framework.dSYM/Contents/Resources/DWARF/fmwk",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data_arm64.linkmap",
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data_x86_64.linkmap",
            "fmwk_arm64.linkmap",
            "fmwk_x86_64.linkmap",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        expected_direct_dsyms = [
            "dSYMs/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM",
            "dSYMs/fmwk.framework.dSYM",
            "dSYMs/fmwk_min_os_baseline_with_bundle.framework.dSYM",
            "dSYMs/fmwk_no_version.framework.dSYM",
            "dSYMs/fmwk_with_resources.framework.dSYM",
        ],
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_no_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_no_bundle_id",
        expected_error = """
Error: There are no attributes set on this target that can be used to determine a bundle ID.

Need a `bundle_id` or a reference to an `apple_base_bundle_id` target coming from the rule or (when
applicable) exactly one of the `apple_capability_set` targets found within `shared_capabilities`.
""",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_empty_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_empty_bundle_id",
        expected_error = """
Error: There are no attributes set on this target that can be used to determine a bundle ID.

Need a `bundle_id` or a reference to an `apple_base_bundle_id` target coming from the rule or (when
applicable) exactly one of the `apple_capability_set` targets found within `shared_capabilities`.
""",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_just_dot_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_just_dot_bundle_id",
        expected_error = "Empty segment in bundle_id: \".\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_leading_dot_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_leading_dot_bundle_id",
        expected_error = "Empty segment in bundle_id: \".my.bundle.id\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_trailing_dot_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_trailing_dot_bundle_id",
        expected_error = "Empty segment in bundle_id: \"my.bundle.id.\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_double_dot_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_double_dot_bundle_id",
        expected_error = "Empty segment in bundle_id: \"my..bundle.id\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_invalid_character_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_invalid_character_bundle_id",
        expected_error = "Invalid character(s) in bundle_id: \"my#bundle\"",
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_custom_bundle_id_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_custom_bundle_id_suffix_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.bundle-id-suffix",
        },
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_ambiguous_shared_capabilities_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ambiguous_shared_capabilities_bundle_id",
        expected_error = """
Error: Found a `bundle_id` on the rule along with `shared_capabilities` defining a `base_bundle_id`.

This is ambiguous. Please remove the `bundle_id` from your rule definition, or reference
`shared_capabilities` without a `base_bundle_id`.
""",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_absent_shared_capabilities_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_absent_shared_capabilities_bundle_id",
        expected_error = """
Error: Expected to find a base_bundle_id from exactly one of the assigned shared_capabilities.
Found none.
""",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_conflicting_shared_capabilities_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_conflicting_shared_capabilities_bundle_id",
        expected_error = """
Error: Received conflicting base bundle IDs from more than one assigned Apple shared capability.

Found "com.bazel.app.example" which does not match previously defined "com.altbazel.app.example".
""",
        tags = [name],
    )

    apple_codesigning_dossier_info_provider_test(
        name = "{}_codesigning_dossier_info_provider_test".format(name),
        expected_dossier = "app_dossier.zip",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_with_spaces_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app minimal has several spaces",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/PkgInfo",
            "$BUNDLE_ROOT/app minimal has several spaces",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_tree_artifact_with_spaces_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app minimal has several spaces",
        build_settings = {
            build_settings_labels.use_tree_artifacts_outputs: "True",
        },
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/PkgInfo",
            "$BUNDLE_ROOT/app minimal has several spaces",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_bundle_name_with_spaces_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal_bundle_name_with_spaces",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/PkgInfo",
            "$BUNDLE_ROOT/app minimal bundle name has several spaces",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_tree_artifact_bundle_name_with_spaces_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal_bundle_name_with_spaces",
        build_settings = {
            build_settings_labels.use_tree_artifacts_outputs: "True",
        },
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/PkgInfo",
            "$BUNDLE_ROOT/app minimal bundle name has several spaces",
        ],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
