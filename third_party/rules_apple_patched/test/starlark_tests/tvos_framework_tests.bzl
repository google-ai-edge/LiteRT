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

"""tvos_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:apple_dsym_bundle_info_test.bzl",
    "apple_dsym_bundle_info_test",
)
load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
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

def tvos_framework_test_suite(name):
    """Test suite for tvos_framework.

    Args:
      name: the base name to be used in things created by this macro
    """
    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "fmwk",
            "CFBundleIdentifier": "com.google.example.framework",
            "CFBundleName": "fmwk",
            "CFBundlePackageType": "FMWK",
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

    # Tests that the bundled .framework contains the expected files.
    archive_contents_test(
        name = "{}_contains_expected_files".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk",
        contains = [
            "$BUNDLE_ROOT/fmwk",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Headers/shared.h",
        ],
        tags = [name],
    )

    # Tests that the correct rpath was added at link-time to the framework's binary.
    # The rpath should match the framework bundle name.
    archive_contents_test(
        name = "{}_binary_has_correct_rpath".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk",
        contains = [
            "$BUNDLE_ROOT/fmwk",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Headers/shared.h",
        ],
        binary_test_file = "$BUNDLE_ROOT/fmwk",
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        tags = [name],
    )

    # Tests that a tvos_framework builds fine without any version info
    # since it isn't required.
    infoplist_contents_test(
        name = "{}_plist_test_with_no_version".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk_with_no_version",
        not_expected_keys = [
            "CFBundleVersion",
            "CFBundleShortVersionString",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_exported_symbols_list_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk_dead_stripped",
        binary_test_file = "$BUNDLE_ROOT/fmwk_dead_stripped",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        binary_not_contains_symbols = ["_dontCallMeShared", "_anticipatedDeadCode"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_angle_bracketed_import_in_umbrella_header".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:static_fmwk",
        text_test_file = "$BUNDLE_ROOT/Headers/static_fmwk.h",
        text_test_values = ["#import <static_fmwk/shared.h>"],
        tags = [name],
    )

    # Verifies transitive "runtime" tvos_framework's are propagated to tvos_application bundle, and
    # are not linked against the app binary. Transitive "runtime" frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    #   - `runtime_dep` of an objc_library target.
    archive_contents_test(
        name = "{}_includes_and_does_not_link_transitive_data_tvos_frameworks".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwks_from_objc_swift_libraries_using_data",
        apple_generate_dsym = True,
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/fmwk_with_structured_resources",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/Images/foo.png",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Images/foo.png",
            "$BUNDLE_ROOT/foo.png",
            "$BUNDLE_ROOT/basic.bundle",
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/simple_bundle_library.bundle",
        ],
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles (offset 24)",
            "name @rpath/fmwk_with_structured_resources.framework/fmwk_with_structured_resources (offset 24)",
        ],
        tags = [name],
    )

    # Verify nested "runtime" tvos_framework's from transitive targets get propagated to
    # tvos_application bundle and are not linked to top-level application. Transitive "runtime"
    # frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    #   - `runtime_dep` of an objc_library target.
    archive_contents_test(
        name = "{}_includes_and_does_not_link_nested_transitive_data_tvos_frameworks".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/fmwk_with_structured_resources",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/Images/foo.png",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Images/foo.png",
            "$BUNDLE_ROOT/foo.png",
            "$BUNDLE_ROOT/basic.bundle",
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/simple_bundle_library.bundle",
        ],
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles (offset 24)",
            "name @rpath/fmwk_with_structured_resources.framework/fmwk_with_structured_resources (offset 24)",
        ],
        tags = [name],
    )

    # Verify that both tvos_framework's listed as load time and runtime dependencies
    # are bundled to top-level application, and runtime frameworks are not linked against
    # the top-level application binary. Transitive "runtime" frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    #   - `runtime_dep` of an objc_library target.
    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_transitive_data_tvos_frameworks".format(name),
        build_type = "simulator",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/fmwk_with_structured_resources",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/Images/foo.png",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Images/foo.png",
            "$BUNDLE_ROOT/foo.png",
            "$BUNDLE_ROOT/basic.bundle",
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/simple_bundle_library.bundle",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles (offset 24)",
            "name @rpath/fmwk_with_structured_resources.framework/fmwk_with_structured_resources (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        tags = [name],
    )

    # Verifies shared resources between app and frameworks propagated via 'data' are not deduped,
    # therefore both app and frameworks contain shared resources.
    archive_contents_test(
        name = "{}_bundles_shared_resources_from_app_and_fmwks_with_data_ios_frameworks".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_resources_and_fmwks_with_resources_from_objc_swift_libraries_using_data",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/fmwk_with_resource_bundles",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/fmwk_with_structured_resources",
            "$BUNDLE_ROOT/Images/foo.png",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        not_contains = [
            "$BUNDLE_ROOT/foo.png",
            "$BUNDLE_ROOT/simple_bundle_library.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/Images/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resource_bundles.framework/foo.png",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_structured_resources.framework/simple_bundle_library.bundle",
        ],
        tags = [name],
    )

    # Test dSYM binaries and linkmaps from framework embedded via 'data' are propagated correctly
    # at the top-level tvos_framework rule, and present through the 'dsysms' and 'linkmaps' output
    # groups.
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_dsyms_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "dsyms",
        expected_outputs = [
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM/Contents/Info.plist",
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM/Contents/Resources/DWARF/fmwk_with_fmwks_from_objc_swift_libraries_using_data",
            "fmwk_with_resource_bundles.framework.dSYM/Contents/Info.plist",
            "fmwk_with_resource_bundles.framework.dSYM/Contents/Resources/DWARF/fmwk_with_resource_bundles",
            "fmwk_with_structured_resources.framework.dSYM/Contents/Info.plist",
            "fmwk_with_structured_resources.framework.dSYM/Contents/Resources/DWARF/fmwk_with_structured_resources",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "linkmaps",
        expected_outputs = [
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data_arm64.linkmap",
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data_x86_64.linkmap",
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
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk_with_fmwks_from_objc_swift_libraries_using_data",
        expected_direct_dsyms = [
            "dSYMs/fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM",
            "dSYMs/fmwk_with_resource_bundles.framework.dSYM",
            "dSYMs/fmwk_with_structured_resources.framework.dSYM",
        ],
        tags = [name],
    )

    # Test that if a tvos_framework target depends on a prebuilt static library (i.e.,
    # apple_static_framework_import), that the static library is defined in the tvos_framework.
    binary_contents_test(
        name = "{}_defines_static_library_impl".format(name),
        build_type = "simulator",
        binary_test_architecture = "x86_64",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_static_framework.framework/fmwk_with_imported_static_framework",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_runtime_framework_using_import_static_lib_dep",
        tags = [name],
    )

    # Test that if a tvos_framework target depends on a prebuilt static library (i.e.,
    # apple_static_framework_import), that the static library is NOT defined in its associated
    # tvos_application.
    binary_contents_test(
        name = "{}_associated_tvos_application_does_not_define_static_library_impl".format(name),
        build_type = "simulator",
        binary_test_architecture = "x86_64",
        binary_test_file = "$BINARY",
        binary_not_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_runtime_framework_using_import_static_lib_dep",
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_base_bundle_id_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:fmwk_with_base_bundle_id_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.fmwk-with-base-bundle-id-derived-bundle-id",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
