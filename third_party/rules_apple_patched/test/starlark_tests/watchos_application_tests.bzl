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

"""watchos_application Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
load(
    "//test/starlark_tests/rules:analysis_target_actions_test.bzl",
    "analysis_target_actions_test",
)
load(
    "//test/starlark_tests/rules:apple_verification_test.bzl",
    "apple_verification_test",
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

def watchos_application_test_suite(name):
    """Test suite for watchos_application.

    Args:
      name: the base name to be used in things created by this macro
    """
    apple_verification_test(
        name = "{}_codesign_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app",
        verifier_script = "verifier_scripts/codesign_verifier.sh",
        tags = [name],
    )

    apple_verification_test(
        name = "{}_no_custom_fmwks_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_with_ext_with_imported_fmwk",
        verifier_script = "verifier_scripts/no_custom_fmwks_verifier.sh",
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "app",
            "CFBundleIdentifier": "com.google.example",
            "CFBundleName": "app",
            "CFBundlePackageType": "APPL",
            "CFBundleSupportedPlatforms:0": "WatchSimulator*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "watchsimulator",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "watchsimulator*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_watchos.baseline,
            "UIDeviceFamily:0": "4",
            "WKWatchKitApp": "true",
        },
        tags = [name],
    )

    # Tests xcasset tool is passed the correct arguments.
    analysis_target_actions_test(
        name = "{}_xcasset_actool_argv".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app",
        target_mnemonic = "AssetCatalogCompile",
        expected_argv = [
            "xctoolrunner actool --compile",
            "--minimum-deployment-target " + common.min_os_watchos.baseline,
            "--platform watchsimulator",
        ],
        tags = [name],
    )

    # Tests that the WatchKit stub executable is bundled everywhere it's
    # supposed to be. This must be tested through the companion app since
    # the `WatchKitSupport2` directory is only added at the root of archives
    # for distribution.
    archive_contents_test(
        name = "{}_contains_stub_executable_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        contains = [
            "$ARCHIVE_ROOT/WatchKitSupport2/WK",
            "$BUNDLE_ROOT/Watch/app.app/_WatchKitStub/WK",
        ],
        tags = [name],
    )

    # Test that the output stub binary is identified as watchOS simulator via the Mach-O load
    # command LC_VERSION_MIN_WATCHOS for the arm64 binary slice when only iOS cpus are defined, and
    # that 32-bit archs are eliminated.
    binary_contents_test(
        name = "{}_simulator_ios_cpus_intel_platform_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        cpus = {
            "ios_multi_cpus": ["x86_64", "sim_arm64"],
            "watchos_cpus": [""],
        },
        binary_test_file = "$BUNDLE_ROOT/Watch/app.app/_WatchKitStub/WK",
        binary_test_architecture = "x86_64",
        binary_not_contains_architectures = ["i386", "arm64e"],
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_WATCHOS"],
        tags = [name],
    )

    # Test that the output application binary is identified as watchOS simulator via the Mach-O
    # load command LC_BUILD_VERSION for the arm64 binary slice when only iOS cpus are defined, and
    # that 32-bit archs are eliminated.
    binary_contents_test(
        name = "{}_simulator_ios_cpus_arm_platform_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        cpus = {
            "ios_multi_cpus": ["x86_64", "sim_arm64"],
            "watchos_cpus": [""],
        },
        binary_test_file = "$BUNDLE_ROOT/Watch/app.app/_WatchKitStub/WK",
        binary_test_architecture = "arm64",
        binary_not_contains_architectures = ["i386", "arm64e"],
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform WATCHOSSIMULATOR"],
        tags = [name],
    )

    # Test that the output stub binary is identified as watchOS simulator via the Mach-O load
    # command LC_VERSION_MIN_WATCHOS for the arm64 binary slice when only iOS cpus are defined, and
    # that it defaults to armv7k.
    binary_contents_test(
        name = "{}_device_ios_cpus_platform_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        cpus = {
            "ios_multi_cpus": ["arm64"],
            "watchos_cpus": [""],
        },
        binary_test_file = "$BUNDLE_ROOT/Watch/app.app/app",
        binary_test_architecture = "armv7k",
        binary_not_contains_architectures = ["arm64_32", "arm64", "arm64e"],
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_WATCHOS"],
        tags = [name],
    )

    # Test that the output multi-arch stub binary is identified as watchOS simulator via the Mach-O
    # load command LC_BUILD_VERSION for the arm64 binary slice, and that 32-bit archs are
    # eliminated.
    binary_contents_test(
        name = "{}_simulator_multiarch_platform_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        cpus = {
            "watchos_cpus": ["x86_64", "arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/Watch/app.app/_WatchKitStub/WK",
        binary_test_architecture = "arm64",
        binary_not_contains_architectures = ["i386", "arm64e"],
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform WATCHOSSIMULATOR"],
        tags = [name],
    )

    # Test that the output multi-arch stub binary is identified as watchOS device via the Mach-O
    # load command LC_VERSION_MIN_WATCHOS for the arm64_32 binary slice, and that 64-bit archs are
    # eliminated.
    binary_contents_test(
        name = "{}_device_multiarch_arm32_platform_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        cpus = {
            "watchos_cpus": ["armv7k", "arm64_32"],
        },
        binary_test_file = "$BUNDLE_ROOT/Watch/app.app/_WatchKitStub/WK",
        binary_not_contains_architectures = ["arm64e", "arm64"],
        binary_test_architecture = "arm64_32",
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_WATCHOS"],
        tags = [name],
    )

    # Test that the output binary for a single arch build is identified as watchOS device via the
    # Mach-O load command LC_VERSION_MIN_WATCHOS for the arm64_32 binary slice, and that the 64-bit
    # archs and the armv7k arch are eliminated.
    binary_contents_test(
        name = "{}_device_arm64_32_platform_test".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_companion",
        cpus = {
            "watchos_cpus": ["arm64_32"],
        },
        binary_test_file = "$BUNDLE_ROOT/Watch/app.app/_WatchKitStub/WK",
        binary_not_contains_architectures = ["armv7k", "arm64e", "arm64"],
        binary_test_architecture = "arm64_32",
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_WATCHOS"],
        tags = [name],
    )

    # Tests inclusion of extensions within Watch extensions
    archive_contents_test(
        name = "{}_contains_watchos_extension_extension".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ios_watchos_with_watchos_extension",
        contains = [
            "$BUNDLE_ROOT/Watch/app.app/PlugIns/ext.appex/PlugIns/watchos_app_extension.appex/watchos_app_extension",
        ],
        tags = [name],
    )

    # Tests inclusion of extensions within Watch extensions if defined with `extensions` as opposed to `extension`
    archive_contents_test(
        name = "{}_contains_watchos_extension_extensions".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ios_watchos_with_watchos_extension_within_extensions",
        contains = [
            "$BUNDLE_ROOT/Watch/app.app/PlugIns/ext.appex/PlugIns/watchos_app_extension.appex/watchos_app_extension",
        ],
        tags = [name],
    )

    # Tests that the tsan support libraries are found in the app extension bundle of a watchOS app.
    archive_contents_test(
        name = "{}_contains_tsan_dylib_device_test".format(name),
        build_type = "simulator",
        cpus = {
            # Thread sanitizer support does not exist for the 32 bit Intel simulator; force the
            # build to be 64 bit to get around this issue.
            "watchos_cpus": ["x86_64"],
        },
        contains = [
            "$BUNDLE_ROOT/Frameworks/libclang_rt.tsan_iossim_dynamic.dylib",
            "$BUNDLE_ROOT/Watch/app.app/PlugIns/ext.appex/Frameworks/libclang_rt.tsan_watchossim_dynamic.dylib",
        ],
        sanitizer = "tsan",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ios_watchos_with_watchos_extension",
        tags = [name],
    )

    # Test app with App Intents generates and bundles Metadata.appintents bundle.
    archive_contents_test(
        name = "{}_contains_app_intents_metadata_bundle".format(name),
        build_type = "simulator",
        cpus = {"watchos_cpus": ["arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_with_app_intents",
        contains = [
            "$BUNDLE_ROOT/Metadata.appintents/extract.actionsdata",
            "$BUNDLE_ROOT/Metadata.appintents/version.json",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_with_ext_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.watchkitapp",
        },
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_test_watchos_single_target_application_required_error".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:app_with_ext_with_invalid_watchos_version",
        expected_error = "Error: Building an app extension-based watchOS 2 application for watchOS 9.0 or later.",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
