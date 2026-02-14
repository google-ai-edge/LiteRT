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

"""visionos_application Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_output_group_info_files_test.bzl",
    "analysis_output_group_info_files_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_actions_test.bzl",
    "analysis_target_actions_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_outputs_test.bzl",
    "analysis_target_outputs_test",
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

visibility("private")

def visionos_application_test_suite(name):
    """Test suite for visionos_application.

    Args:
      name: the base name to be used in things created by this macro
    """

    analysis_target_outputs_test(
        name = "{}_default_app_bundle_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        expected_outputs = ["app.app"],
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_bundle_contents_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/app",
            "$BUNDLE_ROOT/Assets.car",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/PkgInfo",
        ],
        binary_test_file = "$BUNDLE_ROOT/app",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        tags = [
            name,
            "needs-xcode-latest-beta",
        ],
    )

    archive_contents_test(
        name = "{}_contains_solidstack_images_test".format(name),
        build_type = "simulator",
        contains = ["$BUNDLE_ROOT/Assets.car"],
        text_test_file = "$BUNDLE_ROOT/Assets.car",
        text_test_values = ["Bazel_logo.png"],
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_binary_contents_arm_simulator_platform_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:swift_app",
        cpus = {
            "visionos_cpus": ["sim_arm64"],
        },
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_visionos.baseline, "platform VISIONOSSIMULATOR"],
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_opt_resources_simulator_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        is_binary_plist = [
            "$BUNDLE_ROOT/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_dbg_resources_device_test".format(name),
        build_type = "device",
        compilation_mode = "dbg",
        is_binary_plist = ["$BUNDLE_ROOT/resource_bundle.bundle/Info.plist"],
        is_not_binary_plist = ["$BUNDLE_ROOT/Another.plist"],
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_opt_resources_device_test".format(name),
        build_type = "device",
        compilation_mode = "opt",
        is_binary_plist = [
            "$BUNDLE_ROOT/resource_bundle.bundle/Info.plist",
            "$BUNDLE_ROOT/Another.plist",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_opt_strings_simulator_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        contains = [
            "$RESOURCE_ROOT/localization.bundle/en.lproj/files.stringsdict",
            "$RESOURCE_ROOT/localization.bundle/en.lproj/greetings.strings",
        ],
        tags = [
            name,
        ],
    )

    analysis_output_group_info_files_test(
        name = "{}_dsyms_output_group_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        output_group_name = "dsyms",
        expected_outputs = [
            "app.app.dSYM/Contents/Info.plist",
            "app.app.dSYM/Contents/Resources/DWARF/app",
        ],
        tags = [
            name,
        ],
    )

    apple_dsym_bundle_info_test(
        name = "{}_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        expected_direct_dsyms = [
            "dSYMs/app.app.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/app.app.dSYM",
        ],
        tags = [
            name,
        ],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "app",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "app",
            "CFBundlePackageType": "APPL",
            "CFBundleSupportedPlatforms:0": "XR*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "xr*",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "xr*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_visionos.baseline,
            "UIDeviceFamily:0": "7",
        },
        tags = [
            name,
        ],
    )

    # Tests that the linkmap outputs are produced when `--objc_generate_linkmap`
    # is present.
    linkmap_test(
        name = "{}_linkmap_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        architectures = ["arm64"],
        tags = [
            name,
        ],
    )

    analysis_output_group_info_files_test(
        name = "{}_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        output_group_name = "linkmaps",
        expected_outputs = [
            "app_arm64.linkmap",
        ],
        tags = [
            name,
        ],
    )

    # Tests that the provisioning profile is present when built for device.
    archive_contents_test(
        name = "{}_contains_provisioning_profile_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        contains = [
            "$BUNDLE_ROOT/embedded.mobileprovision",
        ],
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_contains_asan_dylib_simulator_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.asan_xrossim_dynamic.dylib",
        ],
        sanitizer = "asan",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app_minimal",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_contains_asan_dylib_device_test".format(name),
        build_type = "device",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.asan_xros_dynamic.dylib",
        ],
        sanitizer = "asan",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app_minimal",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_contains_tsan_dylib_device_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.tsan_xrossim_dynamic.dylib",
        ],
        sanitizer = "tsan",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app_minimal",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_contains_ubsan_dylib_simulator_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.ubsan_xrossim_dynamic.dylib",
        ],
        sanitizer = "ubsan",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app_minimal",
        tags = [
            name,
        ],
    )

    archive_contents_test(
        name = "{}_contains_ubsan_dylib_device_test".format(name),
        build_type = "device",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.ubsan_xros_dynamic.dylib",
        ],
        sanitizer = "ubsan",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app_minimal",
        tags = [
            name,
        ],
    )

    apple_codesigning_dossier_info_provider_test(
        name = "{}_codesigning_dossier_info_provider_test".format(name),
        expected_dossier = "app_dossier.zip",
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        tags = [
            name,
        ],
    )

    # Test that visionOS apps include the --app-icon argument in actool command.
    analysis_target_actions_test(
        name = "{}_includes_app_icon_in_actool_command".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
        target_mnemonic = "AssetCatalogCompile",
        expected_argv = ["--app-icon AppIcon"],
        tags = [name],
    )

    # TODO(b/288582842): Support an IPA output via this output group. This will require some changes
    # to bundling, as the bundle-first build goes through a different set of Python tooling.
    #output_group_zip_contents_test(
    #    name = "{}_has_combined_zip_output_group".format(name),
    #    build_type = "device",
    #    target_under_test = "//test/starlark_tests/targets_under_test/visionos:app",
    #    output_group_name = "combined_dossier_zip",
    #    output_group_file_shortpath = "third_party/bazel_rules/rules_apple/test/starlark_tests/targets_under_test/visionos/app_dossier_with_bundle.zip",
    #    contains = [
    #        "bundle/Payload/app.app/Info.plist",
    #        "bundle/Payload/app.app/app",
    #        "dossier/manifest.json",
    #    ],
    #    tags = [
    #        name,
    #        "needs-xcode-latest-beta",
    #    ],
    #)

    native.test_suite(
        name = name,
        tags = [name],
    )
