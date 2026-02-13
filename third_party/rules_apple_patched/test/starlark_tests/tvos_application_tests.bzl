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

"""tvos_application Starlark tests."""

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
    "analysis_target_actions_test",
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

def tvos_application_test_suite(name):
    """Test suite for tvos_application.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_fmwk_in_fmwk_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_fmwk_in_fmwk_provisioned_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk_provisioned",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_two_fmwk_provisioned_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_two_fmwk_provisioned",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_imported_fmwk_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_imported_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    # Tests that Swift standard libraries bundled in SwiftSupport have the code
    # signature from Apple.
    archive_contents_test(
        name = "{}_swift_support_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_direct_swift_dep",
        binary_test_file = "$ARCHIVE_ROOT/SwiftSupport/appletvos/libswiftCore.dylib",
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
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_entitlements_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_resources_simulator_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/launch_screen_tvos.storyboardc/",
            "$BUNDLE_ROOT/resource_bundle.bundle/Info.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_resources_device_test".format(name),
        build_type = "device",
        contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/launch_screen_tvos.storyboardc/",
            "$BUNDLE_ROOT/resource_bundle.bundle/Info.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_strings_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        contains = [
            "$RESOURCE_ROOT/localization.bundle/en.lproj/files.stringsdict",
            "$RESOURCE_ROOT/localization.bundle/en.lproj/greetings.strings",
        ],
        tags = [name],
    )

    # Tests xcasset tool is passed the correct arguments.
    analysis_target_actions_test(
        name = "{}_xcasset_actool_argv".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        target_mnemonic = "AssetCatalogCompile",
        expected_argv = [
            "xctoolrunner actool --compile",
            "--minimum-deployment-target " + common.min_os_tvos.baseline,
            "--platform appletvsimulator",
        ],
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        output_group_name = "dsyms",
        expected_outputs = [
            "app.app.dSYM/Contents/Info.plist",
            "app.app.dSYM/Contents/Resources/DWARF/app",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
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
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk_provisioned",
        output_group_name = "dsyms",
        expected_outputs = [
            "app_with_fmwk_with_fmwk_provisioned.app.dSYM/Contents/Info.plist",
            "app_with_fmwk_with_fmwk_provisioned.app.dSYM/Contents/Resources/DWARF/app_with_fmwk_with_fmwk_provisioned",
            "fmwk_with_provisioning.framework.dSYM/Contents/Info.plist",
            "fmwk_with_provisioning.framework.dSYM/Contents/Resources/DWARF/fmwk_with_provisioning",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_transitive_dsyms_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk_provisioned",
        expected_direct_dsyms = [
            "dSYMs/app_with_fmwk_with_fmwk_provisioned.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/fmwk_with_provisioning.framework.dSYM",
            "dSYMs/app_with_fmwk_with_fmwk_provisioned.app.dSYM",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "app",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "app",
            "CFBundlePackageType": "APPL",
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
            "UILaunchStoryboardName": "launch_screen_tvos",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_multiple_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_multiple_infoplists",
        expected_values = {
            "AnotherKey": "AnotherValue",
            "CFBundleExecutable": "app_multiple_infoplists",
        },
        tags = [name],
    )

    # Tests that the linkmap outputs are produced when `--objc_generate_linkmap`
    # is present.
    linkmap_test(
        name = "{}_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_x86_64.linkmap",
            "app_arm64.linkmap",
        ],
        tags = [name],
    )

    # Tests that the provisioning profile is present when built for device.
    archive_contents_test(
        name = "{}_contains_provisioning_profile_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        contains = [
            "$BUNDLE_ROOT/embedded.mobileprovision",
        ],
        tags = [name],
    )

    # Tests analysis phase failure when an extension depends on a framework which
    # is not marked extension_safe.
    analysis_failure_message_test(
        name = "{}_fails_with_extension_depending_on_not_extension_safe_framework".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_ext_with_fmwk_not_extension_safe",
        expected_error = (
            "The target {target} is for an extension but its framework dependency " +
            "{framework} is not marked extension-safe. Specify 'extension_safe " +
            "= True' on the framework target."
        ).format(
            framework = Label("//test/starlark_tests/targets_under_test/tvos:fmwk_not_extension_safe"),
            target = Label("//test/starlark_tests/targets_under_test/tvos:ext_with_fmwk_not_extension_safe"),
        ),
        tags = [name],
    )

    # Test that if a tvos_framework target depends on a prebuilt framework (i.e.,
    # apple_dynamic_framework_import), that the inner framework is propagated up
    # to the application and not nested in the outer framework.
    archive_contents_test(
        name = "{}_contains_framework_depends_on_prebuilt_apple_framework_import".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_runtime_framework_using_import_framework_dep",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_dynamic_framework.framework/fmwk_with_imported_dynamic_framework",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_dynamic_framework.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/generated_tvos_dynamic_fmwk.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/generated_tvos_dynamic_fmwk.framework/Resources/generated_tvos_dynamic_fmwk.bundle/Info.plist",
            "$BUNDLE_ROOT/Frameworks/generated_tvos_dynamic_fmwk.framework/generated_tvos_dynamic_fmwk",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Frameworks/generated_tvos_dynamic_fmwk.framework/Frameworks/fmwk_with_imported_dynamic_framework.framework/",
        ],
        tags = [name],
    )

    # Tests that the bundled application contains the framework but that the
    # extension inside it does *not* contain another copy.
    archive_contents_test(
        name = "{}_contains_framework_and_framework_depending_extension_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_framework_and_framework_depending_ext",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Headers/shared.h",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/PlugIns/ext_with_framework.appex/Info.plist",
            "$BUNDLE_ROOT/PlugIns/ext_with_framework.appex/ext_with_framework",
        ],
        not_contains = [
            "$BUNDLE_ROOT/PlugIns/ext_with_framework.appex/Frameworks/",
        ],
        tags = [name],
    )

    # Tests that resources that both apps and frameworks depend on are present
    # in the .framework directory and that the symbols are only present in the
    # framework binary.
    archive_contents_test(
        name = "{}_with_resources_and_framework_resources_contains_files_only_on_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_transitive_structured_resources",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_transitive_structured_resources.framework/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_transitive_structured_resources.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_transitive_structured_resources.framework/fmwk_with_transitive_structured_resources",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Images/foo.png",
        ],
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_transitive_structured_resources.framework/fmwk_with_transitive_structured_resources",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_dontCallMeShared",
            "_anotherFunctionShared",
            "_anticipatedDeadCode",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_with_resources_and_framework_resources_app_binary_not_contains_symbols".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_transitive_structured_resources",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwk_with_transitive_structured_resources",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = [
            "_dontCallMeShared",
            "_anotherFunctionShared",
            "_anticipatedDeadCode",
        ],
        tags = [name],
    )

    # Tests that a framework is present in the top level application
    # bundle in the case that only extensions depend on the framework
    # and the application itself does not.
    archive_contents_test(
        name = "{}_propagates_framework_from_tvos_extension_and_not_bundles_framework_on_extension".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_framework_depending_ext",
        contains = [
            # The main bundle should contain the framework...
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Headers/shared.h",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/PlugIns/ext_with_framework.appex/Info.plist",
            "$BUNDLE_ROOT/PlugIns/ext_with_framework.appex/ext_with_framework",
        ],
        not_contains = [
            # The extension bundle should be intact, but have no inner framework.
            "$BUNDLE_ROOT/PlugIns/ext_with_framework.appex/Frameworks/",
        ],
        tags = [name],
    )

    # Tests that resource bundles that are dependencies of a framework are
    # bundled with the framework if no deduplication is happening.
    archive_contents_test(
        name = "{}_contains_resource_bundles_in_framework_and_not_in_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_resource_bundles",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/simple_bundle_library.bundle/generated.strings",
        ],
        not_contains = [
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        tags = [name],
    )

    # Tests that an App->Framework->Framework dependency is handled properly. (That
    # a framework that is not directly depended on by the app is still pulled into
    # the app, and symbols end up in the correct binaries.)
    archive_contents_test(
        name = "{}_contains_shared_framework_resource_files_only_in_inner_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        contains = [
            # Contains expected framework files...
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/Info.plist",
            # Contains expected shared framework resource file...
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Images/foo.png",
        ],
        not_contains = [
            # Doesn't contains shared framework resource file...
            "$BUNDLE_ROOT/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/Images/foo.png",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_depending_fmwk_with_fmwk_shared_lib_symbols_in_inner_framework_binary".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["-[ObjectiveCCommonClass doSomethingCommon]"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_depending_fmwk_with_fmwk_shared_lib_symbols_not_in_outer_framework_binary".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["-[ObjectiveCCommonClass doSomethingCommon]"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_depending_fmwk_with_fmwk_shared_lib_symbols_not_in_app_binary".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwk_with_fmwk",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["-[ObjectiveCCommonClass doSomethingCommon]"],
        tags = [name],
    )

    # They all have Info.plists with the right bundle ids (even though the
    # frameworks share a comment infoplists entry for it).
    # They also all share a common file to add a custom key, ensure that
    # isn't duped away because of the overlap.
    archive_contents_test(
        name = "{}_depending_fmwk_with_fmwk_app_plist_content".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "CFBundleIdentifier": "com.google.example",
            "AnotherKey": "AnotherValue",
        },
        tags = [name],
    )
    archive_contents_test(
        name = "{}_depending_fmwk_with_fmwk_outer_framework_plist_content".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        plist_test_file = "$BUNDLE_ROOT/Frameworks/fmwk.framework/Info.plist",
        plist_test_values = {
            "CFBundleIdentifier": "com.google.example.framework",
            "AnotherKey": "AnotherValue",
        },
        tags = [name],
    )
    archive_contents_test(
        name = "{}_depending_fmwk_with_fmwk_inner_framework_plist_content".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwk_with_fmwk",
        plist_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/Info.plist",
        plist_test_values = {
            "CFBundleIdentifier": "com.google.example.frameworkception",
            "AnotherKey": "AnotherValue",
        },
        tags = [name],
    )

    # Verifies that, when an extension depends on a framework with different
    # minimum_os, symbol subtraction still occurs.
    archive_contents_test(
        name = "{}_with_ext_min_os_nplus1_extension_binary_not_contains_lib_symbols".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_ext_min_os_nplus1",
        binary_test_file = "$BUNDLE_ROOT/PlugIns/ext_min_os_nplus1.appex/ext_min_os_nplus1",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_with_ext_min_os_nplus1_framework_binary_contains_lib_symbols".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_ext_min_os_nplus1",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    # Tests that different root-level resources with the same name are not
    # deduped between framework and app.
    archive_contents_test(
        name = "{}_does_not_dedup_structured_resources_from_framework_and_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_structured_resources_and_fmwk_with_structured_resources",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/Images/foo.png",
            "$BUNDLE_ROOT/Images/foo.png",
        ],
        tags = [name],
    )

    # Tests that root-level resources depended on by both an application and its
    # framework end up in both bundles given that both bundles have explicit owners
    # on the resources
    archive_contents_test(
        name = "{}_contains_root_level_resource_smart_dedupe_resources".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_ext_and_fmwk_with_common_structured_resources",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/Images/foo.png",
            "$BUNDLE_ROOT/Images/foo.png",
            "$BUNDLE_ROOT/PlugIns/ext_with_framework_with_structured_resources.appex/Images/foo.png",
        ],
        tags = [name],
    )

    # Verifies that resource bundles that are dependencies of a framework are
    # bundled with the framework if no deduplication is happening.
    # tvOS application and framework have the same minimum os version.
    archive_contents_test(
        name = "{}_does_not_contain_common_resource_bundle_from_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_resource_bundles_and_fmwk_with_resource_bundles",
        # Assert that the framework contains the bundled files...
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/simple_bundle_library.bundle",
        ],
        # ...and that the application doesn't.
        not_contains = [
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        tags = [name],
    )

    # Verifies that resource bundles that are dependencies of a framework are
    # bundled with the framework if no deduplication is happening.
    # tvOS application has baseline minimum os version and framework has baseline plus one.
    archive_contents_test(
        name = "{}_does_not_contain_common_resource_bundle_from_framework_nplus1".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_resource_bundles_and_fmwk_with_resource_bundles_nplus1",
        # Assert that the framework contains the bundled files...
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles_nplus1.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles_nplus1.framework/simple_bundle_library.bundle",
        ],
        # ...and that the application doesn't.
        not_contains = [
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        tags = [name],
    )

    # Test app with App Intents generates and bundles Metadata.appintents bundle.
    archive_contents_test(
        name = "{}_contains_app_intents_metadata_bundle".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_app_intents",
        contains = [
            "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
            "$BUNDLE_ROOT/Metadata.appintents/version.json",
        ],
        tags = [name],
    )

    # Test dSYM binaries and linkmaps from framework embedded via 'data' are propagated correctly
    # at the top-level tvos_application rule, and present through the 'dsysms' and 'linkmaps' output
    # groups.
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_dsyms_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        output_group_name = "dsyms",
        expected_outputs = [
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM/Contents/Info.plist",
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM/Contents/Resources/DWARF/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
            # Frameworks
            "fmwk.framework.dSYM/Contents/Info.plist",
            "fmwk.framework.dSYM/Contents/Resources/DWARF/fmwk",
            "fmwk_with_resource_bundles.framework.dSYM/Contents/Info.plist",
            "fmwk_with_resource_bundles.framework.dSYM/Contents/Resources/DWARF/fmwk_with_resource_bundles",
            "fmwk_with_structured_resources.framework.dSYM/Contents/Info.plist",
            "fmwk_with_structured_resources.framework.dSYM/Contents/Resources/DWARF/fmwk_with_structured_resources",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data_arm64.linkmap",
            "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data_x86_64.linkmap",
            "fmwk_arm64.linkmap",
            "fmwk_x86_64.linkmap",
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
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        expected_direct_dsyms = [
            "dSYMs/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data.app.dSYM",
            "dSYMs/fmwk.framework.dSYM",
            "dSYMs/fmwk_with_resource_bundles.framework.dSYM",
            "dSYMs/fmwk_with_structured_resources.framework.dSYM",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example",
        },
        tags = [name],
    )

    # Test that tvOS apps include the --app-icon argument in actool command.
    analysis_target_actions_test(
        name = "{}_includes_app_icon_in_actool_command".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app",
        target_mnemonic = "AssetCatalogCompile",
        expected_argv = ["--app-icon TVBrandAssets"],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
