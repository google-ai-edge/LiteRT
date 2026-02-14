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
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
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
    ":common.bzl",
    "common",
)

def apple_static_xcframework_test_suite(name):
    """Test suite for apple_static_xcframework.

    Args:
      name: the base name to be used in things created by this macro
    """
    archive_contents_test(
        name = "{}_ios_root_plist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework",
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "AvailableLibraries:0:LibraryIdentifier": "ios-arm64",
            "AvailableLibraries:0:LibraryPath": "ios_static_xcframework.framework",
            "AvailableLibraries:0:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:0:SupportedPlatform": "ios",
            "AvailableLibraries:1:LibraryIdentifier": "ios-arm64_x86_64-simulator",
            "AvailableLibraries:1:LibraryPath": "ios_static_xcframework.framework",
            "AvailableLibraries:1:SupportedArchitectures:0": "arm64",
            "AvailableLibraries:1:SupportedArchitectures:1": "x86_64",
            "AvailableLibraries:1:SupportedPlatform": "ios",
            "CFBundlePackageType": "XFWK",
            "XCFrameworkFormatVersion": "1.0",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_arm64_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework",
        contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework.framework/Headers/ios_static_xcframework.h",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework.framework/ios_static_xcframework",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework.framework/Headers/shared.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework.framework/Headers/ios_static_xcframework.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework.framework/ios_static_xcframework",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_avoid_deps_test".format(name),
        build_type = "device",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcfmwk_with_avoid_deps",
        contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_avoid_deps.framework/ios_static_xcfmwk_with_avoid_deps",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_avoid_deps.framework/ios_static_xcfmwk_with_avoid_deps",
            "$BUNDLE_ROOT/Info.plist",
        ],
        not_contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_avoid_deps.frameworks/Headers/DummyFmwk.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_avoid_deps.frameworks/Headers/DummyFmwk.h",
        ],
        binary_test_file = "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_avoid_deps.framework/ios_static_xcfmwk_with_avoid_deps",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_doStuff"],
        binary_not_contains_symbols = ["_frameworkDependent"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_avoid_deps_bundles_test".format(name),
        build_type = "device",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_xcframework_bundling_static_fmwks_with_avoid_deps",
        contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_xcframework_bundling_static_fmwks_with_avoid_deps.framework/ios_xcframework_bundling_static_fmwks_with_avoid_deps",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_xcframework_bundling_static_fmwks_with_avoid_deps.framework/ios_xcframework_bundling_static_fmwks_with_avoid_deps",
            "$BUNDLE_ROOT/Info.plist",
        ],
        not_contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_xcframework_bundling_static_fmwks_with_avoid_deps.framework/resource_bundle.bundle",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_xcframework_bundling_static_fmwks_with_avoid_deps.framework/resource_bundle.bundle",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_objc_generated_modulemap_file_content_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcfmwk_with_objc_sdk_dylibs_and_and_sdk_frameworks",
        text_test_file = "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_objc_sdk_dylibs_and_and_sdk_frameworks.framework/Modules/module.modulemap",
        text_test_values = [
            "framework module ios_static_xcfmwk_with_objc_sdk_dylibs_and_and_sdk_frameworks",
            "umbrella header \"ios_static_xcfmwk_with_objc_sdk_dylibs_and_and_sdk_frameworks.h\"",
            "link \"c++\"",
            "link \"sqlite3\"",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_swift_ios_arm64_x86_64_archive_contents_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcfmwk_with_swift",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_swift.framework/Modules/ios_static_xcfmwk_with_swift.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_swift.framework/ios_static_xcfmwk_with_swift",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift.framework/Modules/ios_static_xcfmwk_with_swift.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift.framework/Modules/ios_static_xcfmwk_with_swift.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift.framework/ios_static_xcfmwk_with_swift",
        ],
        tags = [name],
    )

    # Test that the Swift generated header is propagated to the Headers directory visible within
    # this iOS statix XCFramework along with the Swift interfaces and modulemap files.
    archive_contents_test(
        name = "{}_swift_generates_header_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcfmwk_with_swift_generated_headers",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_swift_generated_headers.framework/Headers/ios_static_xcfmwk_with_swift_generated_headers.h",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_swift_generated_headers.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_swift_generated_headers.framework/Modules/ios_static_xcfmwk_with_swift_generated_headers.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_swift_generated_headers.framework/ios_static_xcfmwk_with_swift_generated_headers",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift_generated_headers.framework/Headers/ios_static_xcfmwk_with_swift_generated_headers.h",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift_generated_headers.framework/Modules/module.modulemap",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift_generated_headers.framework/Modules/ios_static_xcfmwk_with_swift_generated_headers.swiftmodule/arm64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift_generated_headers.framework/Modules/ios_static_xcfmwk_with_swift_generated_headers.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_swift_generated_headers.framework/ios_static_xcfmwk_with_swift_generated_headers",
        ],
        tags = [name],
    )

    # Tests below verify device/simulator builds for static libraries using Mach-O load commands.
    # Logic behind which load command gets written, and platform information can be found on LLVM's:
    #     - llvm/include/llvm/BinaryFormat/MachO.h
    #     - llvm/llvm-project/llvm/lib/MC/MCStreamer.cpp

    # Verify device/simulator static libraries with Mach-O load commands:
    #   - LC_VERSION_MIN_IOS: Present if target minimum version is below 12.0 and is not arm64 sim.
    #   - LC_BUILD_VERSION: Present if target minimum version is above 12.0 or is arm64 sim.
    archive_contents_test(
        name = "{}_ios_arm64_macho_load_cmd_for_simulator".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_oldest_supported",
        binary_test_architecture = "arm64",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework_oldest_supported.framework/ios_static_xcframework_oldest_supported",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_ios.arm_sim_support, "platform IOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_IPHONEOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_ios_x86_64_below_12_0_macho_load_cmd_for_simulator".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_oldest_supported",
        binary_test_architecture = "x86_64",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework_oldest_supported.framework/ios_static_xcframework_oldest_supported",
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_IPHONEOS", "version " + common.min_os_ios.oldest_supported],
        macho_load_commands_not_contain = ["cmd LC_BUILD_VERSION"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_ios_x86_64_above_12_0_macho_load_cmd_for_simulator".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework",
        binary_test_architecture = "x86_64",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcframework.framework/ios_static_xcframework",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform IOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_IPHONEOS"],
        tags = [name],
    )

    # Verifies device static libraries build with Mach-O load commands.
    #   - LC_VERSION_MIN_IOS: Present if target minimum version is below 12.0.
    #   - LC_BUILD_VERSION: Present if target minimum version is above 12.0.
    archive_contents_test(
        name = "{}_ios_x86_64_arm64_below_12_0_macho_load_cmd_for_device".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_oldest_supported",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework_oldest_supported.framework/ios_static_xcframework_oldest_supported",
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_IPHONEOS", "version " + common.min_os_ios.oldest_supported],
        macho_load_commands_not_contain = ["cmd LC_BUILD_VERSION"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_ios_x86_64_arm64_above_12_0_macho_load_cmd_for_device".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework",
        binary_test_file = "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework.framework/ios_static_xcframework",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_ios.baseline, "platform IOS"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_IPHONEOS"],
        tags = [name],
    )

    # Verifies that the include scanning feature builds for the given XCFramework rule.
    archive_contents_test(
        name = "{}_ios_arm64_cc_include_scanning_test".format(name),
        build_type = "device",
        target_features = ["cc_include_scanning"],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework",
        contains = [
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework.framework/ios_static_xcframework",
        ],
        tags = [name],
    )

    # Verifies that bundle_name changes the embedded static libraries and the modulemap file as well
    # as the name of the bundle for the xcframeworks.
    archive_contents_test(
        name = "{}_ios_bundle_name_contents_swift_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcfmwk_with_swift_and_bundle_name",
        contains = [
            "$ARCHIVE_ROOT/ios_static_xcfmwk_with_custom_bundle_name.xcframework/",
            "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_custom_bundle_name.framework/ios_static_xcfmwk_with_custom_bundle_name",
            "$BUNDLE_ROOT/ios-arm64_x86_64-simulator/ios_static_xcfmwk_with_custom_bundle_name.framework/ios_static_xcfmwk_with_custom_bundle_name",
        ],
        text_test_file = "$BUNDLE_ROOT/ios-arm64/ios_static_xcfmwk_with_custom_bundle_name.framework/Modules/module.modulemap",
        text_test_values = [
            "framework module ios_static_xcfmwk_with_custom_bundle_name",
            "header \"ios_static_xcfmwk_with_custom_bundle_name.h\"",
            "requires objc",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_custom_umbrella_header_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_umbrella_header",
        text_test_file = "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework_umbrella_header.framework/Modules/module.modulemap",
        text_test_values = [
            "framework module ios_static_xcframework_umbrella_header",
            "umbrella header \"Umbrella.h\"",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_resources_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_with_resources",
        contains = [
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_static_xcframework_with_resources.framework/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/ios-arm64_arm64e/ios_static_xcframework_with_resources.framework/resource_bundle.bundle/custom_apple_resource_info.out",
        ],
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_overreaching_avoid_deps_swift_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_swift_static_xcframework_with_broad_avoid_deps",
        expected_error = "Error: Could not find a Swift module to build a Swift framework. This could be because \"avoid_deps\" is too broadly defined.",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_ios_inner_framework_infoplist_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_with_bundle_id",
        plist_test_file = "$BUNDLE_ROOT/ios-arm64/ios_static_xcframework_with_bundle_id.framework/Info.plist",
        plist_test_values = {
            "CFBundleIdentifier": "com.example.static.framework",
            "CFBundleExecutable": "ios_static_xcframework_with_bundle_id",
            "CFBundleShortVersionString": "2.1",
            "CFBundleVersion": "2.1.0",
            "MinimumOSVersion": common.min_os_ios.baseline,
        },
        tags = [name],
    )

    directory_test(
        name = "{}_ios_static_library_xcframework_tree_artifact_test".format(name),
        build_settings = {
            build_settings_labels.use_tree_artifacts_outputs: "True",
        },
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework",
        expected_directories = {
            "ios_static_xcframework.xcframework": [
                "Info.plist",
                "ios-arm64/ios_static_xcframework.framework/ios_static_xcframework",
                "ios-arm64/ios_static_xcframework.framework/Headers/ios_static_xcframework.h",
                "ios-arm64/ios_static_xcframework.framework/Headers/shared.h",
                "ios-arm64/ios_static_xcframework.framework/Modules/module.modulemap",
                "ios-arm64_x86_64-simulator/ios_static_xcframework.framework/ios_static_xcframework",
                "ios-arm64_x86_64-simulator/ios_static_xcframework.framework/Headers/ios_static_xcframework.h",
                "ios-arm64_x86_64-simulator/ios_static_xcframework.framework/Headers/shared.h",
                "ios-arm64_x86_64-simulator/ios_static_xcframework.framework/Modules/module.modulemap",
            ],
        },
        tags = [name],
    )

    directory_test(
        name = "{}_ios_static_framework_xcframework_tree_artifact_test".format(name),
        build_settings = {
            build_settings_labels.use_tree_artifacts_outputs: "True",
        },
        target_under_test = "//test/starlark_tests/targets_under_test/apple:ios_static_framework_xcframework_with_deps_resource_bundle",
        expected_directories = {
            "ios_static_framework_xcframework_with_deps_resource_bundle.xcframework": [
                "Info.plist",
                "ios-arm64/ios_static_framework_xcframework_with_deps_resource_bundle.framework/ios_static_framework_xcframework_with_deps_resource_bundle",
                "ios-arm64/ios_static_framework_xcframework_with_deps_resource_bundle.framework/Headers/ios_static_framework_xcframework_with_deps_resource_bundle.h",
                "ios-arm64/ios_static_framework_xcframework_with_deps_resource_bundle.framework/Headers/shared.h",
                "ios-arm64/ios_static_framework_xcframework_with_deps_resource_bundle.framework/Modules/module.modulemap",
                "ios-arm64/ios_static_framework_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/custom_apple_resource_info.out",
                "ios-arm64/ios_static_framework_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/Info.plist",
                "ios-x86_64-simulator/ios_static_framework_xcframework_with_deps_resource_bundle.framework/ios_static_framework_xcframework_with_deps_resource_bundle",
                "ios-x86_64-simulator/ios_static_framework_xcframework_with_deps_resource_bundle.framework/Headers/ios_static_framework_xcframework_with_deps_resource_bundle.h",
                "ios-x86_64-simulator/ios_static_framework_xcframework_with_deps_resource_bundle.framework/Headers/shared.h",
                "ios-x86_64-simulator/ios_static_framework_xcframework_with_deps_resource_bundle.framework/Modules/module.modulemap",
                "ios-x86_64-simulator/ios_static_framework_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/custom_apple_resource_info.out",
                "ios-x86_64-simulator/ios_static_framework_xcframework_with_deps_resource_bundle.framework/resource_bundle.bundle/Info.plist",
            ],
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
