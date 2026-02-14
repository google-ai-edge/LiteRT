# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""apple_static_xcframework_import Starlark tests."""

load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)
load(
    "//test/starlark_tests/rules:analysis_target_actions_test.bzl",
    "analysis_contains_xcframework_processor_action_test",
)
load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

def apple_static_xcframework_import_test_suite(name):
    """Test suite for apple_static_xcframework_import.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Verify the dependent app target successfully builds
    archive_contents_test(
        name = "{}_swift_multi_level_static_xcframework".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_swift_multi_level_static_xcframework",
        contains = ["$ARCHIVE_ROOT/Payload"],
        build_type = "simulator",
        tags = [name],
    )

    # Verify ios_application with XCFramework with static library dependency contains symbols and
    # does not bundle anything under Frameworks/
    archive_contents_test(
        name = "{}_ios_application_with_imported_static_xcframework_includes_symbols".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        not_contains = ["$BUNDLE_ROOT/Frameworks/"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_application_with_imported_static_xcframework_includes_symbols_device".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        not_contains = ["$BUNDLE_ROOT/Frameworks/"],
        tags = [name],
    )

    # Verify ios_application with XCFramework with Swift static library dependency contains
    # Objective-C symbols, doesn't bundle XCFramework, and does bundle Swift standard libraries.
    archive_contents_test(
        name = "{}_swift_ios_application_with_imported_static_xcframework_includes_symbols".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        not_contains = ["$BUNDLE_ROOT/Frameworks/generated_static_xcframework_with_headers"],
        tags = [name],
    )

    # Verify ios_application with an imported XCFramework that has a Swift static library
    # contains symbols visible to Objective-C, and bundles Swift standard libraries.
    archive_contents_test(
        name = "{}_swift_with_imported_static_fmwk_contains_symbols_and_bundles_swift_std_libraries".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_swift_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_OBJC_CLASS_$__TtC34generated_swift_static_xcframework11SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_with_imported_swift_static_fmwk_contains_symbols_and_bundles_swift_std_libraries".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_swift_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_OBJC_CLASS_$__TtC34generated_swift_static_xcframework11SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        tags = [name],
    )

    # Verify Swift standard libraries are bundled for an imported XCFramework that has a Swift
    # static library containing no module interface files (.swiftmodule directory) and where the
    # import rule sets `has_swift` = True.
    archive_contents_test(
        name = "{}_swift_with_no_module_interface_files_and_has_swift_attr_enabled_bundles_swift_std_libraries".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_swift_xcframework_with_static_library_without_swiftmodule",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_OBJC_CLASS_$__TtC34generated_swift_static_xcframework11SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        tags = [name],
    )

    # Verify ios_application links correct XCFramework library between simulator and device builds.
    archive_contents_test(
        name = "{}_links_ios_arm64_macho_load_cmd_for_simulator_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        cpus = {"ios_multi_cpus": ["sim_arm64"]},
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform IOSSIMULATOR"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_links_ios_arm64_macho_load_cmd_for_device_test".format(name),
        build_type = "device",
        cpus = {"ios_multi_cpus": ["arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_xcframework_with_static_library",
        binary_test_architecture = "arm64",
        binary_test_file = "$BINARY",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform IOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_links_ios_arm64e_macho_load_cmd_for_device_test".format(name),
        build_type = "device",
        cpus = {"ios_multi_cpus": ["arm64e"]},
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_xcframework_with_static_library",
        binary_test_architecture = "arm64e",
        binary_test_file = "$BINARY",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform IOS"],
        tags = [name],
    )

    # Verify watchos_application links correct XCFramework library for arm64* architectures.
    archive_contents_test(
        name = "{}_links_watchos_arm64_macho_load_cmd_for_simulator_test".format(name),
        build_type = "simulator",
        cpus = {"watchos_cpus": ["arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_with_imported_static_xcframework",
        not_contains = ["$BUNDLE_ROOT/Frameworks"],
        binary_test_file = "$BUNDLE_ROOT/app_with_imported_static_xcframework",
        binary_test_architecture = "arm64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform WATCHOSSIMULATOR"],
    )
    archive_contents_test(
        name = "{}_links_watchos_arm64_32_macho_load_cmd_for_device_test".format(name),
        build_type = "device",
        cpus = {"watchos_cpus": ["arm64_32"]},
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_with_imported_static_xcframework",
        not_contains = ["$BUNDLE_ROOT/Frameworks"],
        binary_test_file = "$BUNDLE_ROOT/app_with_imported_static_xcframework",
        binary_test_architecture = "arm64_32",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform WATCHOS"],
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
    )

    # Verify tvos_application links XCFramework library for device and simulator architectures.
    archive_contents_test(
        name = "{}_links_imported_tvos_xcframework_to_application_device_build".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_imported_static_xcframework",
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_links_imported_tvos_xcframework_to_application_simulator_arm64_build".format(name),
        build_type = "simulator",
        cpus = {"tvos_cpus": ["sim_arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_imported_static_xcframework",
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOSSIMULATOR"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_links_imported_tvos_xcframework_to_application_simulator_x86_64_build".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_imported_static_xcframework",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOSSIMULATOR"],
        tags = [name],
    )

    # Verify ios_application bundles Framework files when using xcframework_processor_tool.
    archive_contents_test(
        name = "{}_ios_application_with_imported_static_xcframework_includes_symbols_with_xcframework_import_tool".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_imported_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        not_contains = ["$BUNDLE_ROOT/Frameworks/"],
        build_settings = {
            build_settings_labels.parse_xcframework_info_plist: "True",
        },
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_with_imported_swift_static_fmwk_contains_symbols_and_bundles_swift_std_libraries_with_xcframework_import_tool".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_app_with_imported_swift_xcframework_with_static_library",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = [
            "_OBJC_CLASS_$__TtC34generated_swift_static_xcframework11SharedClass",
        ],
        contains = ["$BUNDLE_ROOT/Frameworks/libswiftCore.dylib"],
        build_settings = {
            build_settings_labels.parse_xcframework_info_plist: "True",
        },
        tags = [name],
    )

    # Verify XCFramework processor tool action is registered via build setting.
    analysis_contains_xcframework_processor_action_test(
        name = "{}_ios_application_with_imported_static_xcframework_registers_action_with_xcframework_import_tool".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:ios_imported_static_xcframework",
        target_mnemonic = "ProcessXCFrameworkFiles",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
