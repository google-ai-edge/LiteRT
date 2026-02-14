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

"""macos_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
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

def macos_framework_test_suite(name):
    """Test suite for macos_framework.

    Args:
      name: the base name to be used in things created by this macro
    """
    archive_contents_test(
        name = "{}_with_dot_in_name_builds_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_with_dot.dynamic_framework",
        binary_test_file = "$BUNDLE_ROOT/fmwk_with_dot.dynamic_framework",
        contains = [
            "$BUNDLE_ROOT/fmwk_with_dot.dynamic_framework",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "fmwk",
            "CFBundleIdentifier": "com.google.example.framework",
            "CFBundleName": "fmwk",
            "CFBundlePackageType": "FMWK",
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

    archive_contents_test(
        name = "{}_archive_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk",
        binary_test_file = "$BUNDLE_ROOT/fmwk",
        macho_load_commands_contain = ["name @rpath/fmwk.framework/fmwk (offset 24)"],
        contains = [
            "$BUNDLE_ROOT/fmwk",
            "$BUNDLE_ROOT/Headers/common.h",
            "$BUNDLE_ROOT/Info.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_app_load_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_bundle_only_fmwks",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_bundle_only_fmwks",
        macho_load_commands_not_contain = [
            "name @rpath/bundle_only_fmwk.framework/bundle_only_fmwk (offset 24)",
            "name @rpath/generated_macos_dynamic_fmwk.framework/generated_macos_dynamic_fmwk (offset 24)",
        ],
        contains = [
            "$CONTENT_ROOT/Frameworks/bundle_only_fmwk.framework/bundle_only_fmwk",
            "$CONTENT_ROOT/Frameworks/bundle_only_fmwk.framework/nonlocalized.plist",
            "$CONTENT_ROOT/Frameworks/generated_macos_dynamic_fmwk.framework/generated_macos_dynamic_fmwk",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_no_version_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_no_version",
        not_expected_keys = ["CFBundleShortVersionString", "CFBundleVersion"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_extensions_do_not_duplicate_frameworks_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_ext_and_fmwk_provisioned",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
            "$CONTENT_ROOT/Frameworks/fmwk_with_provisioning.framework/Info.plist",
            "$CONTENT_ROOT/PlugIns/ext_with_fmwk_provisioned.appex",
        ],
        not_contains = ["$CONTENT_ROOT/PlugIns/ext_with_fmwk_provisioned.appex/Frameworks"],
        tags = [name],
    )

    # Tests that different root-level resources with the same name are not
    # deduped between framework and app.
    archive_contents_test(
        name = "{}_same_resource_names_not_deduped".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_same_resource_names_as_framework",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_duplicate_resource_names.framework/Another.plist",
            "$CONTENT_ROOT/Resources/Another.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_extensions_framework_propagates_to_app_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_ext_with_fmwk_provisioned",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
            "$CONTENT_ROOT/Frameworks/fmwk_with_provisioning.framework/Info.plist",
            "$CONTENT_ROOT/PlugIns/ext_with_fmwk_provisioned.appex",
        ],
        not_contains = ["$CONTENT_ROOT/PlugIns/ext_with_fmwk_provisioned.appex/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning"],
        tags = [name],
    )

    # Tests that resources that both apps and frameworks depend on are present
    # in the .framework directory and app directory if both have explicit owners
    # for the resources.
    archive_contents_test(
        name = "{}_shared_resources_with_explicit_owners_in_framework_and_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_framework_and_shared_resources",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$CONTENT_ROOT/Resources/Another.plist",
        ],
        tags = [name],
    )

    # Tests that resources that both apps and frameworks depend on are present
    # in the .framework directory only.
    archive_contents_test(
        name = "{}_resources_in_framework_stays_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_framework_and_resources",
        contains = ["$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist"],
        not_contains = ["$CONTENT_ROOT/Resources/another.plist"],
        tags = [name],
    )

    # Tests that libraries that both apps and frameworks depend only have symbols
    # present in the framework.
    archive_contents_test(
        name = "{}_symbols_from_shared_library_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_framework_and_resources",
        binary_test_architecture = "x86_64",
        binary_test_file = "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        binary_contains_symbols = ["_dontCallMeShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_symbols_from_shared_library_not_in_application".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_framework_and_resources",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_framework_and_resources",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_dontCallMeShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_app_includes_transitive_framework_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwk_with_fmwk",
        binary_test_file = "$CONTENT_ROOT/Frameworks/fmwk.framework/fmwk",
        binary_test_architecture = "x86_64",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
            "$CONTENT_ROOT/Frameworks/fmwk_with_fmwk.framework/Info.plist",
            "$CONTENT_ROOT/Frameworks/fmwk.framework/nonlocalized.plist",
            "$CONTENT_ROOT/Frameworks/fmwk.framework/fmwk",
            "$CONTENT_ROOT/Frameworks/fmwk.framework/Info.plist",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_fmwk.framework/Frameworks/",
            "$CONTENT_ROOT/Frameworks/fmwk_with_fmwk.framework/nonlocalized.plist",
            "$CONTENT_ROOT/framework_resources/nonlocalized.plist",
        ],
        binary_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_app_includes_transitive_framework_symbols_not_in_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwk_with_fmwk",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwk_with_fmwk",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_multiple_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_multiple_infoplists",
        expected_values = {
            "AnotherKey": "AnotherValue",
            "CFBundleExecutable": "fmwk_multiple_infoplists",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_exported_symbols_list_stripped_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_stripped",
        binary_test_file = "$BUNDLE_ROOT/fmwk_stripped",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        binary_not_contains_symbols = ["_dontCallMeShared", "_anticipatedDeadCode"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_two_exported_symbols_lists_stripped_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_stripped_two_exported_symbol_lists",
        binary_test_file = "$BUNDLE_ROOT/fmwk_stripped_two_exported_symbol_lists",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared", "_dontCallMeShared"],
        binary_not_contains_symbols = ["_anticipatedDeadCode"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_exported_symbols_list_dead_stripped_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_dead_stripped",
        binary_test_file = "$BUNDLE_ROOT/fmwk_dead_stripped",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        binary_not_contains_symbols = ["_dontCallMeShared", "_anticipatedDeadCode"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_two_exported_symbols_lists_dead_stripped_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:fmwk_dead_stripped_two_exported_symbol_lists",
        binary_test_file = "$BUNDLE_ROOT/fmwk_dead_stripped_two_exported_symbol_lists",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared", "_dontCallMeShared"],
        binary_not_contains_symbols = ["_anticipatedDeadCode"],
        tags = [name],
    )

    # Test that if an macos_framework target depends on a prebuilt framework, that
    # the inner framework is propagated up to the application and not nested in
    # the outer framework.
    archive_contents_test(
        name = "{}_prebuild_framework_propagated_to_application".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_inner_and_outer_fmwk",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_imported_fmwk.framework/fmwk_with_imported_fmwk",
            "$CONTENT_ROOT/Frameworks/macOSDynamicFramework.framework/macOSDynamicFramework",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_with_imported_fmwk.framework/Frameworks",
        ],
        tags = [name],
    )

    # Test that if an macos_framework target depends on a prebuilt static framework,
    # the inner framework is propagated up to the application and not nested in
    # the outer framework.
    archive_contents_test(
        name = "{}_prebuild_static_framework_included_in_outer_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_inner_and_outer_static_fmwk",
        binary_test_file = "$CONTENT_ROOT/Frameworks/fmwk_with_imported_static_fmwk.framework/fmwk_with_imported_static_fmwk",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["-[ObjectiveCSharedClass doSomethingShared]"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_prebuild_static_framework_not_included_in_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_inner_and_outer_static_fmwk",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_inner_and_outer_static_fmwk",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["-[ObjectiveCSharedClass doSomethingShared]"],
        tags = [name],
    )

    # Verifies that, when an extension depends on a framework with different
    # minimum_os, symbol subtraction still occurs.
    archive_contents_test(
        name = "{}_symbols_present_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_min_os_baseline",
        binary_test_file = "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline.framework/fmwk_min_os_baseline",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    # Tests that framework contains swiftinterface when swift_library enable swift interface/library evolution.
    archive_contents_test(
        name = "{}_static_framework_contains_swiftinterface".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:swift_static_framework",
        contains = [
            "$BUNDLE_ROOT/Headers/swift_framework_lib.h",
            "$BUNDLE_ROOT/Modules/swift_framework_lib.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/Modules/swift_framework_lib.swiftmodule/x86_64.swiftinterface",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Modules/swift_framework_lib.swiftmodule/x86_64.swiftmodule",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_angle_bracketed_import_in_umbrella_header".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:objc_static_framework",
        text_test_file = "$BUNDLE_ROOT/Headers/objc_static_framework.h",
        text_test_values = [
            "#import <objc_static_framework/common.h>",
        ],
        tags = [name],
    )

    # Verifies transitive "runtime" macos_framework's are propagated to macos_application bundle, and
    # are not linked against the app binary. Transitive "runtime" frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    archive_contents_test(
        name = "{}_includes_and_does_not_link_transitive_data_macos_frameworks".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwks_from_objc_swift_libraries_using_data",
        apple_generate_dsym = True,
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
            "$CONTENT_ROOT/Resources/Another.plist",
            "$CONTENT_ROOT/Resources/basic.bundle",
        ],
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwks_from_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_includes_and_does_not_link_transitive_data_macos_frameworks_with_tree_artifacts".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwks_from_objc_swift_libraries_using_data",
        apple_generate_dsym = True,
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
        ],
        not_contains = [
            "$CONTENT_ROOT/Resources/Another.plist",
            "$CONTENT_ROOT/Resources/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
        ],
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwks_from_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        tags = [name],
    )

    # Verify nested "runtime" macos_framework's from transitive targets get propagated to
    # macos_application bundle and are not linked to top-level application. Transitive "runtime"
    # frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    archive_contents_test(
        name = "{}_includes_and_does_not_link_nested_transitive_data_macos_frameworks".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        ],
        not_contains = [
            "$CONTENT_ROOT/Resources/Another.plist",
            "$CONTENT_ROOT/Resources/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
        ],
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        tags = [name],
    )

    # Verify that both macos_framework's listed as load time and runtime dependencies
    # are bundled to top-level application, and runtime frameworks are not linked against
    # the top-level application binary. Transitive "runtime" frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_transitive_data_macos_frameworks".format(name),
        build_type = "device",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk.framework/fmwk",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        ],
        not_contains = [
            "$CONTENT_ROOT/Resources/Another.plist",
            "$CONTENT_ROOT/Resources/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        tags = [name],
    )

    # Verifies shared resources between app and frameworks propagated via 'data' are not deduped,
    # therefore both app and frameworks contain shared resources.
    archive_contents_test(
        name = "{}_bundles_shared_resources_from_app_and_fmwks_with_data_macos_frameworks".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_resources_and_fmwks_with_resources_from_objc_swift_libraries_using_data",
        contains = [
            "$CONTENT_ROOT/Resources/Another.plist",
            "$CONTENT_ROOT/Resources/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$CONTENT_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        ],
        not_contains = [
            "$CONTENT_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$CONTENT_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
        ],
        tags = [name],
    )

    # Verifies an macos_framework listed both as a load time dependency, and dependency of another
    # macos_framework target listed as a runtime dependency:
    #
    # - Framework is bundled at the top-level application.
    # - Framework is linked to the application binary.
    # - Framework does not cause duplicate files error due being listed at two different targets:
    #     1. As a load time dependency at the macos_application target using `frameworks`.
    #     2. As a load time dependency to a runtime macos_framework target using `frameworks`.
    #
    # - Runtime framework is bundled at the top-level application.
    # - Runtime framework is not linked to the application binary.
    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_framework_dep_without_duplicate_files_errors".format(name),
        build_type = "device",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwk_and_ext_with_objc_lib_with_nested_macos_framework",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk.framework/fmwk",
            "$CONTENT_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_fmwk.framework/fmwk_with_fmwk (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwk_and_ext_with_objc_lib_with_nested_macos_framework",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_framework_dep_without_duplicate_files_errors_with_tree_artifacts".format(name),
        build_type = "device",
        binary_test_file = "$CONTENT_ROOT/MacOS/app_with_fmwk_and_ext_with_objc_lib_with_nested_macos_framework",
        contains = [
            "$CONTENT_ROOT/Frameworks/fmwk.framework/fmwk",
            "$CONTENT_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_fmwk.framework/fmwk_with_fmwk (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_fmwk_and_ext_with_objc_lib_with_nested_macos_framework",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
