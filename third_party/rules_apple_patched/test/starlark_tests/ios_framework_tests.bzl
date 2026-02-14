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

"""ios_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
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
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def ios_framework_test_suite(name):
    """Test suite for ios_framework.

    Args:
      name: the base name to be used in things created by this macro
    """
    archive_contents_test(
        name = "{}_with_dot_in_name_builds_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_with_dot.dynamic_framework",
        binary_test_file = "$BUNDLE_ROOT/fmwk_with_dot.dynamic_framework",
        contains = [
            "$BUNDLE_ROOT/fmwk_with_dot.dynamic_framework",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "fmwk",
            "CFBundleIdentifier": "com.google.example.framework",
            "CFBundleName": "fmwk",
            "CFBundlePackageType": "FMWK",
            "CFBundleSupportedPlatforms:0": "iPhone*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "iphone*",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "iphone*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": common.min_os_ios.baseline,
            "UIDeviceFamily:0": "1",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_archive_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_bundle_only_fmwks",
        binary_test_file = "$BUNDLE_ROOT/app_with_bundle_only_fmwks",
        macho_load_commands_not_contain = [
            "name @rpath/bundle_only_fmwk.framework/bundle_only_fmwk (offset 24)",
            "name @rpath/generated_ios_dynamic_fmwk.framework/generated_ios_dynamic_fmwk (offset 24)",
            "name @rpath/ios_dynamic_xcframework.framework/ios_dynamic_xcframework (offset 24)",
        ],
        contains = [
            "$BUNDLE_ROOT/Frameworks/bundle_only_fmwk.framework/bundle_only_fmwk",
            "$BUNDLE_ROOT/Frameworks/bundle_only_fmwk.framework/nonlocalized.plist",
            "$BUNDLE_ROOT/Frameworks/generated_ios_dynamic_fmwk.framework/generated_ios_dynamic_fmwk",
            "$BUNDLE_ROOT/Frameworks/ios_dynamic_xcframework.framework/ios_dynamic_xcframework",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_no_version_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_no_version",
        not_expected_keys = ["CFBundleShortVersionString", "CFBundleVersion"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_extensions_do_not_duplicate_frameworks_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_and_fmwk_provisioned",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/Info.plist",
            "$BUNDLE_ROOT/PlugIns/ext_with_fmwk_provisioned.appex",
        ],
        not_contains = ["$BUNDLE_ROOT/PlugIns/ext_with_fmwk_provisioned.appex/Frameworks"],
        tags = [name],
    )

    # Tests that different root-level resources with the same name are not
    # deduped between framework and app.
    archive_contents_test(
        name = "{}_same_resource_names_not_deduped".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_same_resource_names_as_framework",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_duplicate_resource_names.framework/Another.plist",
            "$BUNDLE_ROOT/Another.plist",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_extensions_framework_propagates_to_app_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_with_fmwk_provisioned",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_provisioning.framework/Info.plist",
            "$BUNDLE_ROOT/PlugIns/ext_with_fmwk_provisioned.appex",
        ],
        not_contains = ["$BUNDLE_ROOT/PlugIns/ext_with_fmwk_provisioned.appex/Frameworks/fmwk_with_provisioning.framework/fmwk_with_provisioning"],
        tags = [name],
    )

    # Tests that if frameworks have resource bundles they are only in the
    # framework.
    archive_contents_test(
        name = "{}_resource_bundle_in_framework_stays_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_with_bundle_resources",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_nplus1.framework/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_nplus1.framework/basic.bundle/nested/should_be_nested.strings",
        ],
        not_contains = [
            "$BUNDLE_ROOT/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/basic.bundle/nested/should_be_nested.strings",
        ],
        tags = [name],
    )

    # Tests that if frameworks and applications have different minimum versions
    # the assets are still only in the framework.
    archive_contents_test(
        name = "{}_resources_in_framework_stays_in_framework_with_app_with_lower_min_os_version".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_baseline_min_os_and_nplus1_fmwk",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_nplus1.framework/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_nplus1.framework/basic.bundle/nested/should_be_nested.strings",
        ],
        not_contains = [
            "$BUNDLE_ROOT/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/basic.bundle/nested/should_be_nested.strings",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_resources_in_framework_stays_in_framework_with_app_with_higher_min_os_version".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_nplus1_min_os_and_baseline_fmwk",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle/nested/should_be_nested.strings",
        ],
        not_contains = [
            "$BUNDLE_ROOT/basic.bundle/basic_bundle.txt",
            "$BUNDLE_ROOT/basic.bundle/nested/should_be_nested.strings",
        ],
        tags = [name],
    )

    # Tests that resources that both apps and frameworks depend on are present
    # in the .framework directory and app directory if both have explicit owners
    # for the resources.
    archive_contents_test(
        name = "{}_shared_resources_with_explicit_owners_in_framework_and_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_framework_and_shared_resources",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$BUNDLE_ROOT/Another.plist",
        ],
        tags = [name],
    )

    # Tests that resources that both apps and frameworks depend on are present
    # in the .framework directory only.
    archive_contents_test(
        name = "{}_resources_in_framework_stays_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_framework_and_resources",
        contains = ["$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist"],
        not_contains = ["$BUNDLE_ROOT/another.plist"],
        tags = [name],
    )

    # Tests that libraries that both apps and frameworks depend only have symbols
    # present in the framework.
    archive_contents_test(
        name = "{}_symbols_from_shared_library_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_framework_and_resources",
        binary_test_architecture = "x86_64",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        binary_contains_symbols = ["_dontCallMeShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_symbols_from_shared_library_not_in_application".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_framework_and_resources",
        binary_test_file = "$BUNDLE_ROOT/app_with_framework_and_resources",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_dontCallMeShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_app_includes_transitive_framework_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_with_fmwk",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
        binary_test_architecture = "x86_64",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/Info.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/nonlocalized.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/Info.plist",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/Frameworks/",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/nonlocalized.plist",
            "$BUNDLE_ROOT/framework_resources/nonlocalized.plist",
        ],
        binary_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_app_includes_transitive_framework_symbols_not_in_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_with_fmwk",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwk_with_fmwk",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_multiple_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_multiple_infoplists",
        expected_values = {
            "AnotherKey": "AnotherValue",
            "CFBundleExecutable": "fmwk_multiple_infoplists",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_exported_symbols_list_stripped_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_stripped",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_stripped_two_exported_symbol_lists",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_dead_stripped",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_dead_stripped_two_exported_symbol_lists",
        binary_test_file = "$BUNDLE_ROOT/fmwk_dead_stripped_two_exported_symbol_lists",
        compilation_mode = "opt",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared", "_dontCallMeShared"],
        binary_not_contains_symbols = ["_anticipatedDeadCode"],
        tags = [name],
    )

    # Test that if an ios_framework target depends on a prebuilt framework, that
    # the inner framework is propagated up to the application and not nested in
    # the outer framework.
    archive_contents_test(
        name = "{}_prebuild_framework_propagated_to_application".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_inner_and_outer_fmwk",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_fmwk.framework/fmwk_with_imported_fmwk",
            "$BUNDLE_ROOT/Frameworks/iOSDynamicFramework.framework/iOSDynamicFramework",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_fmwk.framework/Frameworks",
        ],
        tags = [name],
    )

    # Test that if an ios_framework target depends on a prebuilt static framework,
    # the inner framework is propagated up to the application and not nested in
    # the outer framework.
    archive_contents_test(
        name = "{}_prebuild_static_framework_included_in_outer_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_inner_and_outer_static_fmwk",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_with_imported_static_fmwk.framework/fmwk_with_imported_static_fmwk",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["-[ObjectiveCSharedClass doSomethingShared]"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_prebuild_static_framework_not_included_in_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_inner_and_outer_static_fmwk",
        binary_test_file = "$BUNDLE_ROOT/app_with_inner_and_outer_static_fmwk",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["-[ObjectiveCSharedClass doSomethingShared]"],
        tags = [name],
    )

    # Verifies that, when an extension depends on a framework with different
    # minimum_os, symbol subtraction still occurs.
    archive_contents_test(
        name = "{}_symbols_present_in_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_min_os_baseline",
        binary_test_file = "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline.framework/fmwk_min_os_baseline",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_symbols_not_in_extension".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_min_os_baseline",
        binary_test_file = "$BUNDLE_ROOT/PlugIns/ext_min_os_nplus1.appex/ext_min_os_nplus1",
        binary_test_architecture = "x86_64",
        binary_not_contains_symbols = ["_anotherFunctionShared"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_static_framework_contains_swiftinterface".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:swift_static_framework_with_swiftinterface",
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
        target_under_test = "//test/starlark_tests/targets_under_test/ios:objc_static_framework",
        text_test_file = "$BUNDLE_ROOT/Headers/objc_static_framework.h",
        text_test_values = [
            "#import <objc_static_framework/common.h>",
        ],
        tags = [name],
    )

    # Verifies transitive "runtime" ios_framework's are propagated to ios_application bundle, and
    # are not linked against the app binary. Transitive "runtime" frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    archive_contents_test(
        name = "{}_includes_and_does_not_link_transitive_data_ios_frameworks".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_objc_swift_libraries_using_data",
        apple_generate_dsym = True,
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_includes_and_does_not_link_transitive_data_ios_frameworks_with_tree_artifacts".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_objc_swift_libraries_using_data",
        apple_generate_dsym = True,
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        tags = [name],
    )

    # Verify nested "runtime" ios_framework's from transitive targets get propagated to
    # ios_application bundle and are not linked to top-level application. Transitive "runtime"
    # frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    archive_contents_test(
        name = "{}_includes_and_does_not_link_nested_transitive_data_ios_frameworks".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        tags = [name],
    )

    # Verify that both ios_framework's listed as load time and runtime dependencies
    # are bundled to top-level application, and runtime frameworks are not linked against
    # the top-level application binary. Transitive "runtime" frameworks included are:
    #   - `data` of an objc_library target.
    #   - `data` of an swift_library target.
    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_transitive_data_ios_frameworks".format(name),
        build_type = "device",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_no_version.framework/fmwk_no_version (offset 24)",
            "name @rpath/fmwk_with_resources.framework/fmwk_with_resources (offset 24)",
            "name @rpath/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
        tags = [name],
    )

    # Verifies shared resources between app and frameworks propagated via 'data' are not deduped,
    # therefore both app and frameworks contain shared resources.
    archive_contents_test(
        name = "{}_bundles_shared_resources_from_app_and_fmwks_with_data_ios_frameworks".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_resources_and_fmwks_with_resources_from_objc_swift_libraries_using_data",
        contains = [
            "$BUNDLE_ROOT/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/basic.bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/fmwk_min_os_baseline_with_bundle",
            "$BUNDLE_ROOT/Frameworks/fmwk_no_version.framework/fmwk_no_version",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/fmwk_with_resources",
            "$BUNDLE_ROOT/basic.bundle",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk_min_os_baseline_with_bundle.framework/Another.plist",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_resources.framework/basic.bundle",
        ],
        tags = [name],
    )

    # Verifies an ios_framework listed both as a load time dependency, and dependency of another
    # ios_framework target listed as a runtime dependency:
    #
    # - Framework is bundled at the top-level application.
    # - Framework is linked to the application binary.
    # - Framework does not cause duplicate files error due being listed at two different targets:
    #     1. As a load time dependency at the ios_application target using `frameworks`.
    #     2. As a load time dependency to a runtime ios_framework target using `frameworks`.
    #
    # - Runtime framework is bundled at the top-level application.
    # - Runtime framework is not linked to the application binary.
    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_framework_dep_without_duplicate_files_errors".format(name),
        build_type = "device",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwk_and_ext_with_objc_lib_with_nested_ios_framework",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_fmwk.framework/fmwk_with_fmwk (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_and_ext_with_objc_lib_with_nested_ios_framework",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_bundles_both_load_and_runtime_framework_dep_without_duplicate_files_errors_with_tree_artifacts".format(name),
        build_type = "device",
        binary_test_file = "$BUNDLE_ROOT/app_with_fmwk_and_ext_with_objc_lib_with_nested_ios_framework",
        contains = [
            "$BUNDLE_ROOT/Frameworks/fmwk.framework/fmwk",
            "$BUNDLE_ROOT/Frameworks/fmwk_with_fmwk.framework/fmwk_with_fmwk",
        ],
        macho_load_commands_contain = [
            "name @rpath/fmwk.framework/fmwk (offset 24)",
        ],
        macho_load_commands_not_contain = [
            "name @rpath/fmwk_with_fmwk.framework/fmwk_with_fmwk (offset 24)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_fmwk_and_ext_with_objc_lib_with_nested_ios_framework",
        tags = [name],
    )

    # Test dSYM binaries and linkmaps from framework embedded via 'data' are propagated correctly
    # at the top-level ios_framework rule, and present through the 'dsysms' and 'linkmaps' output
    # groups.
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_dsyms_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "dsyms",
        expected_outputs = [
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM/Contents/Info.plist",
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM/Contents/Resources/DWARF/fmwk_with_fmwks_from_objc_swift_libraries_using_data",
            "fmwk_min_os_baseline_with_bundle.framework.dSYM/Contents/Info.plist",
            "fmwk_min_os_baseline_with_bundle.framework.dSYM/Contents/Resources/DWARF/fmwk_min_os_baseline_with_bundle",
            "fmwk_no_version.framework.dSYM/Contents/Info.plist",
            "fmwk_no_version.framework.dSYM/Contents/Resources/DWARF/fmwk_no_version",
            "fmwk_with_resources.framework.dSYM/Contents/Info.plist",
            "fmwk_with_resources.framework.dSYM/Contents/Resources/DWARF/fmwk_with_resources",
        ],
        tags = [name],
    )
    analysis_output_group_info_files_test(
        name = "{}_with_runtime_framework_transitive_linkmaps_output_group_info_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_with_fmwks_from_objc_swift_libraries_using_data",
        output_group_name = "linkmaps",
        expected_outputs = [
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data_arm64.linkmap",
            "fmwk_with_fmwks_from_objc_swift_libraries_using_data_x86_64.linkmap",
            "fmwk_min_os_baseline_with_bundle_arm64.linkmap",
            "fmwk_min_os_baseline_with_bundle_x86_64.linkmap",
            "fmwk_no_version_arm64.linkmap",
            "fmwk_no_version_x86_64.linkmap",
            "fmwk_with_resources_arm64.linkmap",
            "fmwk_with_resources_x86_64.linkmap",
        ],
        tags = [name],
    )

    # Test transitive frameworks dSYM bundles are propagated by the AppleDsymBundleInfo provider.
    apple_dsym_bundle_info_test(
        name = "{}_with_runtime_framework_dsym_bundle_info_files_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_with_fmwks_from_objc_swift_libraries_using_data",
        expected_direct_dsyms = [
            "dSYMs/fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM",
        ],
        expected_transitive_dsyms = [
            "dSYMs/fmwk_with_fmwks_from_objc_swift_libraries_using_data.framework.dSYM",
            "dSYMs/fmwk_min_os_baseline_with_bundle.framework.dSYM",
            "dSYMs/fmwk_no_version.framework.dSYM",
            "dSYMs/fmwk_with_resources.framework.dSYM",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_base_bundle_id_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_with_base_bundle_id_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.fmwk-with-base-bundle-id-derived-bundle-id",
        },
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_ambiguous_base_bundle_id_derived_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:fmwk_with_ambiguous_base_bundle_id_derived_bundle_id",
        expected_error = """
Error: Found a `bundle_id` provided with `base_bundle_id`. This is ambiguous.

Please remove one of the two from your rule definition.
""",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
