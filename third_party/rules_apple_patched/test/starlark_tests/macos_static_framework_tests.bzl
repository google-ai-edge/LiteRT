# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""macos_static_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def macos_static_framework_test_suite(name):
    """Test suite for macos_static_framework.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Tests Swift macos_static_framework builds correctly for sim_arm64, and x86_64 cpu's.
    archive_contents_test(
        name = "{}_swift_arm64_builds_using_macos_cpus".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:swift_macos_static_framework",
        cpus = {
            "macos_cpus": ["arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/SwiftFmwk",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_macos.arm64_support, "platform MACOS"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_MACOSX"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_swift_x86_64_and_arm64_builds_using_macos_cpus".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:swift_macos_static_framework",
        cpus = {
            "macos_cpus": ["x86_64", "arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/SwiftFmwk",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_macos.arm64_support, "platform MACOS"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_MACOSX"],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_swift_x86_64_builds".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:swift_macos_static_framework",
        cpus = {
            "macos_cpus": ["x86_64"],
        },
        binary_test_file = "$BUNDLE_ROOT/SwiftFmwk",
        binary_test_architecture = "x86_64",
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_MACOSX"],
        macho_load_commands_not_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_macos.baseline, "platform MACOS"],
        tags = [name],
    )

    # Test that it's permitted for a static framework to have multiple
    # `swift_library` dependencies if only one module remains after excluding
    # the transitive closure of `avoid_deps`. Likewise, make sure that the
    # symbols from the avoided deps aren't linked in. (In a situation like
    # this, the user must provide those dependencies separately if they are
    # needed.)
    archive_contents_test(
        name = "{}_swift_avoid_deps_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_fmwk_with_swift_and_avoid_deps",
        contains = [
            "$BUNDLE_ROOT/Modules/SwiftFmwkUpperLib.swiftmodule/x86_64.swiftdoc",
        ],
        binary_test_file = "$BUNDLE_ROOT/SwiftFmwkUpperLib",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["_$s17SwiftFmwkUpperLib5DummyVMn"],
        binary_not_contains_symbols = [
            "_$s17SwiftFmwkLowerLib5DummyVMn",
            "_$s18SwiftFmwkLowestLib5DummyVMn",
        ],
        tags = [name],
    )

    # Test that no module map is generated if the target does not have headers
    # and does not depend on any system dylibs/frameworks.
    archive_contents_test(
        name = "{}_no_module_map_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:objc_static_framework_without_modulemap",
        not_contains = ["$BUNDLE_ROOT/Modules/module.modulemap"],
        tags = [name],
    )

    # Test that a module map is generated if the target depends on system
    # dylibs.
    archive_contents_test(
        name = "{}_module_map_with_sdk_dylibs_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:objc_static_framework_with_sdk_dylibs_dep",
        contains = ["$BUNDLE_ROOT/Modules/module.modulemap"],
        tags = [name],
        text_test_file = "$BUNDLE_ROOT/Modules/module.modulemap",
        text_test_values = [" link \"z\""],
    )

    # Test that a module map is generated if the target depends on system
    # frameworks.
    archive_contents_test(
        name = "{}_module_map_with_sdk_fmwks_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:objc_static_framework_with_sdk_fmwks_dep",
        contains = ["$BUNDLE_ROOT/Modules/module.modulemap"],
        tags = [name],
        text_test_file = "$BUNDLE_ROOT/Modules/module.modulemap",
        text_test_values = [" link framework \"CoreData\""],
    )

    # Test that a module map is generated if the target's `hdrs` is not empty.
    archive_contents_test(
        name = "{}_module_map_with_hdrs_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:objc_static_framework",
        contains = ["$BUNDLE_ROOT/Modules/module.modulemap"],
        tags = [name],
        text_test_file = "$BUNDLE_ROOT/Modules/module.modulemap",
        text_test_values = [" umbrella header \"objc_static_framework.h\""],
    )

    # Test that the Swift generated header is propagated to the Headers visible
    # within this macOS framework along with the swift interfaces and modulemap.
    archive_contents_test(
        name = "{}_swift_generates_header_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_framework_with_generated_header",
        contains = [
            "$BUNDLE_ROOT/Headers/SwiftFmwkWithGenHeader.h",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/Modules/SwiftFmwkWithGenHeader.swiftmodule/x86_64.swiftdoc",
        ],
        tags = [name],
    )

    # Tests that attempting to generate dSYMs does not cause the build to fail
    # (apple_static_library does not generate dSYMs, and the bundler should not
    # unconditionally assume that the provider will be present).
    archive_contents_test(
        name = "{}_builds_with_dsyms_enabled_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:swift_macos_static_framework",
        apple_generate_dsym = True,
        contains = ["$BUNDLE_ROOT/SwiftFmwk"],
        tags = [name],
    )

    # Verifies that bundle_name attribute changes the embedded static library, Clang module map,
    # and the name of the framework bundle.
    archive_contents_test(
        name = "{}_bundle_name_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_framework_with_generated_header",
        contains = [
            "$ARCHIVE_ROOT/SwiftFmwkWithGenHeader.framework/SwiftFmwkWithGenHeader",
        ],
        text_test_file = "$BUNDLE_ROOT/Modules/module.modulemap",
        text_test_values = [
            "module SwiftFmwkWithGenHeader",
            "header \"SwiftFmwkWithGenHeader.h\"",
        ],
        tags = [name],
    )

    # Tests sdk_dylib and sdk_framework attributes are captured into the modulemap.
    archive_contents_test(
        name = "{}_generated_modulemap_file_content_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_framework_from_objc_library",
        text_test_file = "$BUNDLE_ROOT/Modules/module.modulemap",
        text_test_values = [
            "link \"c++\"",
            "link \"sqlite3\"",
        ],
        tags = [name],
    )

    # Tests framework bundle does not contain resources when "exclude_resources = True",
    # but does include headers set in the "hdrs" attribute.
    archive_contents_test(
        name = "{}_excludes_transitive_resources_and_contains_header_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_framework_with_header_and_exclude_resources",
        contains = [
            "$BUNDLE_ROOT/static_framework_with_header_and_exclude_resources",
            "$BUNDLE_ROOT/Headers/shared.h",
            "$BUNDLE_ROOT/Modules/module.modulemap",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Assets.car",
            "$BUNDLE_ROOT/unversioned_datamodel.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/v1.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/v2.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/VersionInfo.plist",
            "$BUNDLE_ROOT/storyboard_macos.storyboardc/",
            "$BUNDLE_ROOT/nonlocalized.strings",
            "$BUNDLE_ROOT/view_macos.nib",
        ],
        tags = [name],
    )

    # Tests framework bundle does not contain resources and headers
    # when "exclude_resources = True" and "hdrs" is not set.
    archive_contents_test(
        name = "{}_excludes_transitive_resources_and_header_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_framework_with_no_header_and_exclude_resources",
        contains = [
            "$BUNDLE_ROOT/static_framework_with_no_header_and_exclude_resources",
        ],
        not_contains = [
            "$BUNDLE_ROOT/Headers/shared.h",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Assets.car",
            "$BUNDLE_ROOT/unversioned_datamodel.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/v1.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/v2.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/VersionInfo.plist",
            "$BUNDLE_ROOT/storyboard_macos.storyboardc/",
            "$BUNDLE_ROOT/nonlocalized.strings",
            "$BUNDLE_ROOT/view_macos.nib",
        ],
        tags = [name],
    )

    # Tests framework bundle contains the expected transitive resources
    # when "exclude_resources = False".
    archive_contents_test(
        name = "{}_includes_transitive_resources_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:static_framework_with_transitive_resources",
        contains = [
            "$BUNDLE_ROOT/static_framework_with_transitive_resources",
            "$BUNDLE_ROOT/Headers/shared.h",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/Assets.car",
            "$BUNDLE_ROOT/unversioned_datamodel.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/v1.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/v2.mom",
            "$BUNDLE_ROOT/versioned_datamodel.momd/VersionInfo.plist",
            "$BUNDLE_ROOT/storyboard_macos.storyboardc/",
            "$BUNDLE_ROOT/nonlocalized.strings",
        ],
        not_contains = ["$BUNDLE_ROOT/Info.plist"],
        asset_catalog_test_file = "$BUNDLE_ROOT/Assets.car",
        asset_catalog_test_contains = ["star"],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
