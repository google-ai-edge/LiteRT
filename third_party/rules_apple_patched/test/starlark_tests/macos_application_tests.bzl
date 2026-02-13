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

"""macos_application Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_with_tree_artifact_outputs_test",
)
load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:analysis_runfiles_test.bzl",
    "analysis_runfiles_dsym_test",
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
    "apple_symbols_file_test",
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

def macos_application_test_suite(name):
    """Test suite for macos_application.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_imported_fmwk_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_entitlements_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        verifier_script = "verifier_scripts/entitlements_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_imported_versioned_fmwk_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_versioned_fmwk",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_imported_versioned_xcframework_codesign_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_dynamic_versioned_xcframework",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_swift_dylibs_no_static_linkage_test".format(name),
        build_type = "device",
        contains = [
            "$CONTENT_ROOT/Frameworks/libswiftCore.dylib",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_basic_swift",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_basic_swift",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_swift_slowAlloc"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_additional_contents_test".format(name),
        build_type = "device",
        contains = [
            "$CONTENT_ROOT/Additional/additional.txt",
            "$CONTENT_ROOT/Nested/non_nested.txt",
            "$CONTENT_ROOT/Nested/nested/nested.txt",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_correct_rpath_header_value_test".format(name),
        build_type = "device",
        binary_test_file = "$CONTENT_ROOT/MacOS/app",
        macho_load_commands_contain = ["path @executable_path/../Frameworks (offset 12)"],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_resources_test".format(name),
        build_type = "device",
        contains = [
            "$CONTENT_ROOT/Resources/resource_bundle.bundle/Info.plist",
            "$CONTENT_ROOT/Resources/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_strings_device_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        contains = [
            "$RESOURCE_ROOT/localization.bundle/en.lproj/files.stringsdict",
            "$RESOURCE_ROOT/localization.bundle/en.lproj/greetings.strings",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_custom_linkopts_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_special_linkopts",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_special_linkopts",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_linkopts_test_main"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_bundle_name_with_space_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_space",
        compilation_mode = "opt",
        contains = [
            "$ARCHIVE_ROOT/app with space.app",
        ],
        not_contains = [
            "$ARCHIVE_ROOT/app.app",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_prebuilt_dynamic_framework_dependency_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_fmwk",
        contains = [
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/generated_macos_dynamic_fmwk",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/Resources/Info.plist",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/Modules/module.modulemap",
        ],
        assert_file_permissions = {
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/Resources": "755",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/Resources/Info.plist": "644",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/generated_macos_dynamic_fmwk": "755",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_prebuilt_dynamic_versioned_framework_dependency_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_versioned_fmwk",
        contains = [
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Resources",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Resources/Info.plist",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/generated_macos_dynamic_versioned_fmwk",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/generated_macos_dynamic_versioned_fmwk",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Modules/module.modulemap",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Modules/module.modulemap",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Modules/module.modulemap",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Resources/Info.plist",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/generated_macos_dynamic_versioned_fmwk",
        ],
        assert_file_permissions = {
            # regular files
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Resources/Info.plist": "644",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/generated_macos_dynamic_versioned_fmwk": "755",
            # symbolic links
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Resources": "120755",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current": "120755",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_versioned_fmwk.framework/generated_macos_dynamic_versioned_fmwk": "120755",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_prebuilt_dynamic_versioned_xcframework_dependency_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_dynamic_versioned_xcframework",
        contains = [
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Resources",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Resources/Info.plist",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/generated_dynamic_macos_versioned_xcframework",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/generated_dynamic_macos_versioned_xcframework",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Modules/module.modulemap",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Modules/module.modulemap",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Headers/SharedClass.h",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Modules/module.modulemap",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Resources/Info.plist",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/generated_dynamic_macos_versioned_xcframework",
        ],
        assert_file_permissions = {
            # regular files
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Resources/Info.plist": "644",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/generated_dynamic_macos_versioned_xcframework": "755",
            # symbolic links
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Resources": "120755",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current": "120755",
            "$CONTENT_ROOT/Frameworks/generated_dynamic_macos_versioned_xcframework.framework/generated_dynamic_macos_versioned_xcframework": "120755",
        },
        tags = [name],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        output_group_name = "dsyms",
        expected_outputs = [
            "app.app.dSYM/Contents/Info.plist",
            "app.app.dSYM/Contents/Resources/DWARF/app",
        ],
        tags = [name],
    )
    apple_dsym_bundle_info_test(
        name = "{}_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        expected_direct_dsyms = [
            "dSYMs/app.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app.app.dSYM",
        ],
        tags = [name],
    )

    analysis_runfiles_dsym_test(
        name = "{}_runfiles_dsym_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        expected_runfiles = [
            "test/starlark_tests/targets_under_test/macos/app.app.dSYM/Contents/Resources/DWARF/app",
            "test/starlark_tests/targets_under_test/macos/app.app.dSYM/Contents/Info.plist",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "app",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "app",
            "CFBundlePackageType": "APPL",
            "CFBundleSupportedPlatforms:0": "MacOSX",
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

    # Tests xcasset tool is passed the correct arguments.
    analysis_target_actions_test(
        name = "{}_xcasset_actool_argv".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app",
        target_mnemonic = "AssetCatalogCompile",
        expected_argv = [
            "xctoolrunner actool --compile",
            "--minimum-deployment-target " + common.min_os_macos.baseline,
            "--platform macosx",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_multiple_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_multiple_infoplists",
        expected_values = {
            "AnotherKey": "AnotherValue",
            "CFBundleExecutable": "app_multiple_infoplists",
        },
        tags = [name],
    )

    # Tests that the archive contains .symbols package files when `include_symbols_in_bundle`
    # is enabled.
    apple_symbols_file_test(
        name = "{}_archive_contains_apple_symbols_files_test".format(name),
        binary_paths = [
            "app_with_ext_and_symbols_in_bundle.app/Contents/MacOS/app_with_ext_and_symbols_in_bundle",
            "app_with_ext_and_symbols_in_bundle.app/Contents/PlugIns/ext.appex/Contents/MacOS/ext",
        ],
        build_type = "device",
        tags = [name],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_ext_and_symbols_in_bundle",
    )

    infoplist_contents_test(
        name = "{}_minimum_os_deployment_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_minimum_deployment_os_version",
        expected_values = {
            "LSMinimumSystemVersion": "11.0",
        },
        tags = [name],
    )

    # Verify importing versioned framework with tree artifacts enabled fails.
    analysis_failure_message_with_tree_artifact_outputs_test(
        name = "{}_fails_with_imported_versioned_framework_and_tree_artifact_outputs".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_imported_versioned_fmwk",
        expected_error = (
            "The apple_dynamic_framework_import rule does not yet support versioned " +
            "frameworks with the experimental tree artifact feature/build setting."
        ),
        tags = [name],
    )

    # Test app with App Intents generates and bundles Metadata.appintents bundle.
    archive_contents_test(
        name = "{}_contains_app_intents_metadata_bundle".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_app_intents",
        contains = [
            "$RESOURCE_ROOT/Metadata.appintents/extract.actionsdata",
            "$RESOURCE_ROOT/Metadata.appintents/version.json",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_archive_contains_ccinfo_deps_dylibs_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_ccinfo_dylib_deps",
        contains = [
            "$CONTENT_ROOT/Frameworks/libmylib_with_rpath.dylib",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_archive_contains_cc_library_with_runfiles_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_cc_library_with_runfiles",
        contains = [
            "$CONTENT_ROOT/Resources/test/starlark_tests/resources/cc_lib_resources/runfile_a.txt",
            "$CONTENT_ROOT/Resources/test/starlark_tests/resources/cc_lib_resources/runfile_b.txt",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_archive_contains_cc_library_with_resources_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_cc_library_with_resources",
        contains = [
            "$CONTENT_ROOT/Resources/resource_a.txt",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_archive_contains_cc_library_suppress_resources_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_cc_library_suppress_resources",
        contains = [
            "$CONTENT_ROOT/Resources/test/starlark_tests/resources/cc_lib_resources/runfile_b.txt",
        ],
        not_contains = [
            # Suppressed resource shouldn't be in either runfile or resource location
            "$CONTENT_ROOT/Resources/test/starlark_tests/resources/cc_lib_resources/suppressed_resource.txt",
            "$CONTENT_ROOT/Resources/suppressed_resource.txt",
        ],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
