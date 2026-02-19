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

"""xcframework Starlark tests."""

load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)
load(
    "//test/starlark_tests/rules:action_command_line_test.bzl",
    "action_command_line_test",
)
load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)
load(
    "//test/starlark_tests/rules:directory_test.bzl",
    "directory_test",
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

def apple_xcframework_test_suite(name):
    """Test suite for apple_xcframework.

    Args:
      name: the base name to be used in things created by this macro
    """
    infoplist_contents_test(
        name = "{}_ios_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        expected_values = {
            "AvailableLibraries:0:LibraryIdentifier": "ios-arm64",
            "AvailableLibraries:0:LibraryPath": "ios_dynamic_xcframework.framework",
            "AvailableLibraries:0:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:0:SupportedPlatform": "ios",
            "AvailableLibraries:1:LibraryIdentifier": "ios-x86_64-simulator",
            "AvailableLibraries:1:LibraryPath": "ios_dynamic_xcframework.framework",
            "AvailableLibraries:1:SupportedArchitectures:0": "x86_64",
            "AvailableLibraries:1:SupportedPlatform": "ios",
            "AvailableLibraries:1:SupportedPlatformVariant": "simulator",
            "CFBundlePackageType": "XFWK",
            "XCFrameworkFormatVersion": "1.0",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_ios_fat_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        expected_values = {
            "AvailableLibraries:0:LibraryIdentifier": "ios-arm64_arm64e",
            "AvailableLibraries:0:LibraryPath": "ios_dynamic_lipoed_xcframework.framework",
            "AvailableLibraries:0:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:0:SupportedArchitectures:1": "arm64e",
            "AvailableLibraries:0:SupportedPlatform": "ios",
            "AvailableLibraries:1:LibraryIdentifier": "ios-arm64_x86_64-simulator",
            "AvailableLibraries:1:LibraryPath": "ios_dynamic_lipoed_xcframework.framework",
            "AvailableLibraries:1:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:1:SupportedArchitectures:1": "x86_64",
            "AvailableLibraries:1:SupportedPlatform": "ios",
            "AvailableLibraries:1:SupportedPlatformVariant": "simulator",
            "CFBundlePackageType": "XFWK",
            "XCFrameworkFormatVersion": "1.0",
        },
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_tvos_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        expected_values = {
            "AvailableLibraries:0:LibraryIdentifier": "ios-arm64",
            "AvailableLibraries:0:LibraryPath": "tvos_dynamic_xcframework.framework",
            "AvailableLibraries:0:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:0:SupportedPlatform": "ios",
            "AvailableLibraries:1:LibraryIdentifier": "ios-x86_64-simulator",
            "AvailableLibraries:1:LibraryPath": "tvos_dynamic_xcframework.framework",
            "AvailableLibraries:1:SupportedArchitectures:0": "x86_64",
            "AvailableLibraries:1:SupportedPlatform": "ios",
            "AvailableLibraries:1:SupportedPlatformVariant": "simulator",
            "AvailableLibraries:2:LibraryIdentifier": "tvos-arm64",
            "AvailableLibraries:2:LibraryPath": "tvos_dynamic_xcframework.framework",
            "AvailableLibraries:2:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:2:SupportedPlatform": "tvos",
            "AvailableLibraries:3:LibraryIdentifier": "tvos-arm64_x86_64-simulator",
            "AvailableLibraries:3:LibraryPath": "tvos_dynamic_xcframework.framework",
            "AvailableLibraries:3:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:3:SupportedArchitectures:1": "x86_64",
            "AvailableLibraries:3:SupportedPlatform": "tvos",
            "AvailableLibraries:3:SupportedPlatformVariant": "simulator",
            "CFBundlePackageType": "XFWK",
            "XCFrameworkFormatVersion": "1.0",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_generated_modulemap_file_content_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        text_test_file = "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/Modules/module.modulemap",
        text_test_values = [
            "framework module ios_dynamic_xcframework",
            "header \"ios_dynamic_xcframework.h\"",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_arm64_device_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = [
            "name @rpath/ios_dynamic_xcframework.framework/ios_dynamic_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/Headers/ios_dynamic_xcframework.h",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_x86_64_sim_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
        binary_test_architecture = "x86_64",
        macho_load_commands_contain = [
            "name @rpath/ios_dynamic_xcframework.framework/ios_dynamic_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        contains = [
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/Headers/ios_dynamic_xcframework.h",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_fat_device_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64_arm64e/ios_dynamic_lipoed_xcframework.framework/ios_dynamic_lipoed_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = [
            "name @rpath/ios_dynamic_lipoed_xcframework.framework/ios_dynamic_lipoed_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        contains = [
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_dynamic_lipoed_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_dynamic_lipoed_xcframework.framework/Headers/ios_dynamic_lipoed_xcframework.h",
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_dynamic_lipoed_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_dynamic_lipoed_xcframework.framework/ios_dynamic_lipoed_xcframework",
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_dynamic_lipoed_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_fat_sim_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_xcframework.framework/ios_dynamic_lipoed_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = [
            "name @rpath/ios_dynamic_lipoed_xcframework.framework/ios_dynamic_lipoed_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        contains = [
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_xcframework.framework/Headers/ios_dynamic_lipoed_xcframework.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_xcframework.framework/ios_dynamic_lipoed_xcframework",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_tvos_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        contains = [
            "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/Headers/tvos_dynamic_xcframework.h",
            "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
            "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/Headers/tvos_dynamic_xcframework.h",
            "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
            "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    # XCFrameworks do not provide a public AppleDsymBundleInfo provider for the following reasons:
    #
    #     - All dSYMs for embedded frameworks are provided in output groups when specified with the
    #         --output_groups=+dsyms option.
    #     - There are no known end users that require the usage of dSYMs from XCFrameworks that
    #         are not already served by the output groups API.
    #     - XCFrameworks can embed dSYM bundles within the XCFramework bundle on a per-library
    #         identifier basis, which is not something that the rules have previously supported as a
    #         debugging experience, and would not be effectively represented through this particular
    #         public provider interface.
    #
    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        output_group_name = "dsyms",
        expected_outputs = [
            "ios_dynamic_xcframework_dsyms/ios_dynamic_xcframework_ios_device.framework.dSYM/Contents/Info.plist",
            "ios_dynamic_xcframework_dsyms/ios_dynamic_xcframework_ios_device.framework.dSYM/Contents/Resources/DWARF/ios_dynamic_xcframework_ios_device",
            "ios_dynamic_xcframework_dsyms/ios_dynamic_xcframework_ios_simulator.framework.dSYM/Contents/Info.plist",
            "ios_dynamic_xcframework_dsyms/ios_dynamic_xcframework_ios_simulator.framework.dSYM/Contents/Resources/DWARF/ios_dynamic_xcframework_ios_simulator",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_fat_frameworks_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        output_group_name = "dsyms",
        expected_outputs = [
            "ios_dynamic_lipoed_xcframework_dsyms/ios_dynamic_lipoed_xcframework_ios_device.framework.dSYM/Contents/Info.plist",
            "ios_dynamic_lipoed_xcframework_dsyms/ios_dynamic_lipoed_xcframework_ios_device.framework.dSYM/Contents/Resources/DWARF/ios_dynamic_lipoed_xcframework_ios_device",
            "ios_dynamic_lipoed_xcframework_dsyms/ios_dynamic_lipoed_xcframework_ios_simulator.framework.dSYM/Contents/Info.plist",
            "ios_dynamic_lipoed_xcframework_dsyms/ios_dynamic_lipoed_xcframework_ios_simulator.framework.dSYM/Contents/Resources/DWARF/ios_dynamic_lipoed_xcframework_ios_simulator",
        ],
        tags = [name],
    )

    linkmap_test(
        name = "{}_device_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        expected_linkmap_names = ["ios_dynamic_xcframework_ios_device"],
        architectures = ["arm64"],
        tags = [name],
    )
    linkmap_test(
        name = "{}_simulator_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        expected_linkmap_names = ["ios_dynamic_xcframework_ios_simulator"],
        architectures = ["x86_64"],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        output_group_name = "linkmaps",
        expected_outputs = [
            "ios_dynamic_xcframework_ios_simulator_x86_64.linkmap",
            "ios_dynamic_xcframework_ios_device_arm64.linkmap",
        ],
        tags = [name],
    )

    linkmap_test(
        name = "{}_fat_device_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        expected_linkmap_names = ["ios_dynamic_lipoed_xcframework_ios_device"],
        architectures = ["arm64", "arm64e"],
        tags = [name],
    )
    linkmap_test(
        name = "{}_fat_simulator_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        expected_linkmap_names = ["ios_dynamic_lipoed_xcframework_ios_simulator"],
        architectures = ["x86_64", "arm64"],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_multiple_architectures_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_xcframework",
        output_group_name = "linkmaps",
        expected_outputs = [
            "ios_dynamic_lipoed_xcframework_ios_device_arm64.linkmap",
            "ios_dynamic_lipoed_xcframework_ios_device_arm64e.linkmap",
            "ios_dynamic_lipoed_xcframework_ios_simulator_arm64.linkmap",
            "ios_dynamic_lipoed_xcframework_ios_simulator_x86_64.linkmap",
        ],
        tags = [name],
    )

    # Tests that minimum os versions values are respected by the embedded frameworks.
    archive_contents_test(
        name = "{}_ios_minimum_os_versions_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        plist_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework.framework/Info.plist",
        plist_test_values = {
            "MinimumOSVersion": common.min_os_ios.baseline,
        },
        tags = [name],
    )

    # Tests that options to override the device family (in this case, exclusively "ipad" for the iOS
    # platform) are respected by the embedded frameworks.
    archive_contents_test(
        name = "{}_ios_exclusively_ipad_device_family_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_exclusively_ipad_device_family",
        plist_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_exclusively_ipad_device_family.framework/Info.plist",
        plist_test_values = {
            "UIDeviceFamily:0": "2",
        },
        tags = [name],
    )

    # Tests that info plist merging is respected by XCFrameworks.
    archive_contents_test(
        name = "{}_multiple_infoplist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_multiple_infoplists",
        plist_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_multiple_infoplists.framework/Info.plist",
        plist_test_values = {
            "AnotherKey": "AnotherValue",
            "CFBundleExecutable": "ios_dynamic_xcframework_multiple_infoplists",
        },
        tags = [name],
    )

    # Tests that resource bundles and files assigned through "data" are respected.
    archive_contents_test(
        name = "{}_dbg_resources_data_test".format(name),
        build_type = "device",
        compilation_mode = "dbg",
        is_binary_plist = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_with_data_resource_bundle.framework/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_with_data_resource_bundle.framework/resource_bundle.bundle/Info.plist",
        ],
        is_not_binary_plist = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_with_data_resource_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_with_data_resource_bundle.framework/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_with_data_resource_bundle",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_opt_resources_data_test".format(name),
        build_type = "device",
        compilation_mode = "opt",
        is_binary_plist = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_with_data_resource_bundle.framework/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_with_data_resource_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_with_data_resource_bundle.framework/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_with_data_resource_bundle.framework/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_with_data_resource_bundle",
        tags = [name],
    )

    # Tests that resource bundles assigned through "deps" are respected.
    archive_contents_test(
        name = "{}_dbg_resources_deps_test".format(name),
        build_type = "device",
        compilation_mode = "dbg",
        is_binary_plist = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/Info.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_with_deps_resource_bundle",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_opt_resources_deps_test".format(name),
        build_type = "device",
        compilation_mode = "opt",
        is_binary_plist = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/Info.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_with_deps_resource_bundle",
        tags = [name],
    )

    # Tests that the exported symbols list works for XCFrameworks.
    archive_contents_test(
        name = "{}_exported_symbols_lists_stripped_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_stripped",
        binary_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_stripped.framework/ios_dynamic_xcframework_stripped",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        binary_not_contains_symbols = ["_dontCallMeShared", "_anticipatedDeadCode"],
        tags = [name],
    )

    # Tests that multiple exported symbols lists works for XCFrameworks.
    archive_contents_test(
        name = "{}_two_exported_symbols_lists_stripped_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_stripped_two_exported_symbols_lists",
        binary_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_stripped_two_exported_symbols_lists.framework/ios_dynamic_xcframework_stripped_two_exported_symbols_lists",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared", "_dontCallMeShared"],
        binary_not_contains_symbols = ["_anticipatedDeadCode"],
        tags = [name],
    )

    # Tests that dead stripping + exported symbols lists works for XCFrameworks just as it does for
    # dynamic frameworks.
    archive_contents_test(
        name = "{}_exported_symbols_list_dead_stripped_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_dead_stripped",
        binary_test_file = "$BUNDLE_ROOT/ios-x86_64-simulator/ios_dynamic_xcframework_dead_stripped.framework/ios_dynamic_xcframework_dead_stripped",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        binary_not_contains_symbols = ["_dontCallMeShared", "_anticipatedDeadCode"],
        tags = [name],
    )

    # Tests that generated swift interfaces work for XCFrameworks when a swift_library is included that
    # enable swift interface/library evolution.
    archive_contents_test(
        name = "{}_swift_interface_generation_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_lipoed_swift_xcframework",
        contains = [
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_swift_xcframework.framework/Modules/ios_dynamic_lipoed_swift_xcframework.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_swift_xcframework.framework/Modules/ios_dynamic_lipoed_swift_xcframework.swiftmodule/arm64.swiftinterface",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_swift_xcframework.framework/Modules/ios_dynamic_lipoed_swift_xcframework.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_swift_xcframework.framework/Modules/ios_dynamic_lipoed_swift_xcframework.swiftmodule/x86_64.swiftinterface",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_swift_xcframework.framework/ios_dynamic_lipoed_swift_xcframework",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_dynamic_lipoed_swift_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_lipoed_swift_xcframework.framework/Modules/ios_dynamic_lipoed_swift_xcframework.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_lipoed_swift_xcframework.framework/Modules/ios_dynamic_lipoed_swift_xcframework.swiftmodule/arm64.swiftinterface",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_lipoed_swift_xcframework.framework/ios_dynamic_lipoed_swift_xcframework",
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_lipoed_swift_xcframework.framework/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    # Test that the Swift generated header is propagated to the Headers visible within this iOS
    # framework along with the swift interfaces and modulemap.
    archive_contents_test(
        name = "{}_swift_generates_header_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_swift_xcframework_with_generated_header",
        contains = [
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/SwiftFmwkWithGenHeader.framework/Headers/SwiftFmwkWithGenHeader.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/SwiftFmwkWithGenHeader.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/SwiftFmwkWithGenHeader.framework/Modules/SwiftFmwkWithGenHeader.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/SwiftFmwkWithGenHeader.framework/Modules/SwiftFmwkWithGenHeader.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64/SwiftFmwkWithGenHeader.framework/Headers/SwiftFmwkWithGenHeader.h",
            "$BUNDLE_ROOT/ios-arm64/SwiftFmwkWithGenHeader.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64/SwiftFmwkWithGenHeader.framework/Modules/SwiftFmwkWithGenHeader.swiftmodule/arm64.swiftdoc",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_bundle_name_contents_swift_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_swift_xcframework_with_generated_header",
        text_test_file = "$BUNDLE_ROOT/ios-arm64/SwiftFmwkWithGenHeader.framework/Modules/module.modulemap",
        text_test_values = [
            "framework module SwiftFmwkWithGenHeader",
            "header \"SwiftFmwkWithGenHeader.h\"",
            "requires objc",
        ],
        tags = [name],
    )

    # Verifies that the include scanning feature builds for the given XCFramework rule.
    archive_contents_test(
        name = "{}_ios_arm64_cc_include_scanning_test".format(name),
        build_type = "device",
        target_features = ["cc_include_scanning"],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_custom_umbrella_header_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_umbrella_header",
        text_test_file = "$BUNDLE_ROOT/ios-arm64/ios_dynamic_xcframework_umbrella_header.framework/Modules/module.modulemap",
        text_test_values = [
            "framework module ios_dynamic_xcframework",
            "header \"Umbrella.h\"",
        ],
        tags = [name],
    )

    # Test tvOS XCFramework binaries contain Mach-O load commands for device or simulator.
    archive_contents_test(
        name = "{}_tvos_simulator_binary_contains_macho_load_cmd_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_TVOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_tvos_device_binary_contains_macho_load_cmd_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOS"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_TVOS"],
        tags = [name],
    )

    # Test tvOS XCFramework binaries have the correct rpaths.
    archive_contents_test(
        name = "{}_tvos_simulator_binary_contains_arm64_rpaths_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = [
            "name @rpath/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_tvos_simulator_binary_contains_x86_64_rpaths_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/tvos-arm64_x86_64-simulator/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
        binary_test_architecture = "x86_64",
        macho_load_commands_contain = [
            "name @rpath/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_tvos_device_binary_contains_rpaths_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:tvos_dynamic_xcframework",
        binary_test_file = "$BUNDLE_ROOT/tvos-arm64/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = [
            "name @rpath/tvos_dynamic_xcframework.framework/tvos_dynamic_xcframework (offset 24)",
            "path @executable_path/Frameworks (offset 12)",
        ],
        tags = [name],
    )

    action_command_line_test(
        name = "{}_xcframework_with_extension_safe".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        mnemonic = "ObjcLink",
        expected_argv = [
            "-fapplication-extension",
        ],
        tags = [name],
    )

    action_command_line_test(
        name = "{}_xcframework_without_extension_safe".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework_umbrella_header",
        mnemonic = "ObjcLink",
        not_expected_argv = [
            "-fapplication-extension",
        ],
        tags = [name],
    )

    directory_test(
        name = "{}_ios_dynamic_xcframework_tree_artifact_test".format(name),
        build_settings = {
            build_settings_labels.use_tree_artifacts_outputs: "True",
        },
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_dynamic_xcframework",
        expected_directories = {
            "ios_dynamic_xcframework.xcframework": [
                "Info.plist",
                "ios-arm64/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
                "ios-arm64/ios_dynamic_xcframework.framework/Info.plist",
                "ios-arm64/ios_dynamic_xcframework.framework/Headers/ios_dynamic_xcframework.h",
                "ios-arm64/ios_dynamic_xcframework.framework/Headers/shared.h",
                "ios-arm64/ios_dynamic_xcframework.framework/Modules/module.modulemap",
                "ios-x86_64-simulator/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
                "ios-x86_64-simulator/ios_dynamic_xcframework.framework/Info.plist",
                "ios-x86_64-simulator/ios_dynamic_xcframework.framework/Headers/ios_dynamic_xcframework.h",
                "ios-x86_64-simulator/ios_dynamic_xcframework.framework/Headers/shared.h",
                "ios-x86_64-simulator/ios_dynamic_xcframework.framework/Modules/module.modulemap",
            ],
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
