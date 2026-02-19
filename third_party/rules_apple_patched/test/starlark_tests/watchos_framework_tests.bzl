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

"""watchos_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
    "binary_contents_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)

def watchos_framework_test_suite(name):
    """Test suite for watchos_framework.

    Args:
      name: the base name to be used in things created by this macro
    """
    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:fmwk",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "fmwk",
            "CFBundleIdentifier": "com.google.example.framework",
            "CFBundleName": "fmwk",
            "CFBundlePackageType": "FMWK",
            "CFBundleSupportedPlatforms:0": "WatchSimulator*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "watchsimulator*",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "watchsimulator*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": "4.0",
            "UIDeviceFamily:0": "4",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_exported_symbols_list_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:fmwk_dead_stripped",
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
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:static_fmwk",
        text_test_file = "$BUNDLE_ROOT/Headers/static_fmwk.h",
        text_test_values = ["#import <static_fmwk/shared.h>"],
        tags = [name],
    )

    # Verify watchos_framework listed as a data of an objc_library gets
    # propagated to watchos_extension bundle.
    archive_contents_test(
        name = "{}_includes_objc_library_watchos_framework_data".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ext_with_objc_library_dep_with_watchos_framework_data",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
        ],
        tags = [name],
    )

    # Verify nested frameworks from objc_library targets get propagated to
    # watchos_extension bundle.
    archive_contents_test(
        name = "{}_includes_multiple_objc_library_watchos_framework_deps".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ext_with_objc_lib_dep_with_inner_lib_with_data_fmwk",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
        ],
        tags = [name],
    )

    # Verify watchos_framework listed as a data of an objc_library does not
    # get linked to top-level extension (Mach-O LC_LOAD_DYLIB commands).
    archive_contents_test(
        name = "{}_does_not_load_bundled_watchos_framework_data".format(name),
        build_type = "simulator",
        binary_test_file = "$BUNDLE_ROOT/ext_with_objc_lib_dep_with_inner_lib_with_data_fmwk",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
            "name @rpath/fmwk_with_provisioning.framework/fmwk_with_provisioning (offset 24)",
            "name @rpath/fmwk_with_fmwk.framework/fmwk_with_fmwk (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ext_with_objc_lib_dep_with_inner_lib_with_data_fmwk",
        tags = [name],
    )

    # Verify that both watchos_framework listed as a load time and data
    # get bundled to top-level extension, and runtime does not get linked.
    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_framework_dep".format(name),
        build_type = "simulator",
        binary_test_file = "$BUNDLE_ROOT/ext_with_load_and_runtime_framework_dep",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_provisioning.framework/fmwk_with_provisioning (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ext_with_load_and_runtime_framework_dep",
        tags = [name],
    )

    # Test that if a watchos_framework target depends on a prebuilt static library (i.e.,
    # apple_static_framework_import), that the static library is defined in the watchos_framework.
    binary_contents_test(
        name = "{}_defines_static_library_impl".format(name),
        build_type = "simulator",
        binary_test_architecture = "i386",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_static_framework.framework/fmwk_with_imported_static_framework",
        binary_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ext_with_runtime_framework_using_import_static_lib_dep",
        tags = [name],
    )

    # Test that if a watchos_framework target depends on a prebuilt static library (i.e.,
    # apple_static_framework_import), that the static library is NOT defined in its associated
    # watchos_extension.
    binary_contents_test(
        name = "{}_associated_watchos_application_does_not_define_static_library_impl".format(name),
        build_type = "simulator",
        binary_test_architecture = "i386",
        binary_test_file = "$BINARY",
        binary_not_contains_symbols = [
            "-[SharedClass doSomethingShared]",
            "_OBJC_CLASS_$_SharedClass",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:ext_with_runtime_framework_using_import_static_lib_dep",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
