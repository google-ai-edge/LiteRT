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

"""xcarchive Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

def xcarchive_test_suite(name):
    """Test suite for xcarchive rule.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Verify xcarchive bundles required files and app for simulator and device.
    archive_contents_test(
        name = "{}_contains_xcarchive_files_simulator".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal.xcarchive",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Products/Applications/app_minimal.app",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "ApplicationProperties:ApplicationPath": "Applications/app_minimal.app",
            "ApplicationProperties:ArchiveVersion": "2",
            "ApplicationProperties:CFBundleIdentifier": "com.google.example",
            "ApplicationProperties:CFBundleShortVersionString": "1.0",
            "ApplicationProperties:CFBundleVersion": "1.0",
            "Name": "app_minimal",
            "SchemeName": "app_minimal",
        },
        tags = [name],
    )
    archive_contents_test(
        name = "{}_contains_xcarchive_files_simulator_dsyms".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal.xcarchive",
        contains = [
            "$BUNDLE_ROOT/dSYMs/app_minimal.app.dSYM",
            "$BUNDLE_ROOT/dSYMs/app_minimal.app.dSYM/Contents/Resources/DWARF/app_minimal",
            "$BUNDLE_ROOT/dSYMs/app_minimal.app.dSYM/Contents/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Products/Applications/app_minimal.app",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "ApplicationProperties:ApplicationPath": "Applications/app_minimal.app",
            "ApplicationProperties:ArchiveVersion": "2",
            "ApplicationProperties:CFBundleIdentifier": "com.google.example",
            "ApplicationProperties:CFBundleShortVersionString": "1.0",
            "ApplicationProperties:CFBundleVersion": "1.0",
            "Name": "app_minimal",
            "SchemeName": "app_minimal",
        },
        apple_generate_dsym = True,
        tags = [name],
    )
    archive_contents_test(
        name = "{}_contains_xcarchive_files_simulator_dsyms_extensions".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_ext_space_in_path.xcarchive",
        contains = [
            "$BUNDLE_ROOT/dSYMs/app_with_ext_space_in_path.app.dSYM",
            "$BUNDLE_ROOT/dSYMs/ext with space.appex.dSYM",
            "$BUNDLE_ROOT/dSYMs/ext with space.appex.dSYM/Contents/Resources/DWARF/ext with space",
            "$BUNDLE_ROOT/dSYMs/ext with space.appex.dSYM/Contents/Info.plist",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Products/Applications/app_with_ext_space_in_path.app",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "ApplicationProperties:ApplicationPath": "Applications/app_with_ext_space_in_path.app",
            "ApplicationProperties:ArchiveVersion": "2",
            "ApplicationProperties:CFBundleIdentifier": "com.google.example",
            "ApplicationProperties:CFBundleShortVersionString": "1.0",
            "ApplicationProperties:CFBundleVersion": "1.0",
            "Name": "app_with_ext_space_in_path",
            "SchemeName": "app_with_ext_space_in_path",
        },
        apple_generate_dsym = True,
        tags = [name],
    )
    archive_contents_test(
        name = "{}_contains_xcarchive_files_simulator_linkmaps".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal.xcarchive",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Linkmaps/app_minimal_x86_64.linkmap",
            "$BUNDLE_ROOT/Products/Applications/app_minimal.app",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "ApplicationProperties:ApplicationPath": "Applications/app_minimal.app",
            "ApplicationProperties:ArchiveVersion": "2",
            "ApplicationProperties:CFBundleIdentifier": "com.google.example",
            "ApplicationProperties:CFBundleShortVersionString": "1.0",
            "ApplicationProperties:CFBundleVersion": "1.0",
            "Name": "app_minimal",
            "SchemeName": "app_minimal",
        },
        objc_generate_linkmap = True,
        tags = [name],
    )
    archive_contents_test(
        name = "{}_contains_xcarchive_files_device".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_minimal.xcarchive",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Products/Applications/app_minimal.app",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "ApplicationProperties:ApplicationPath": "Applications/app_minimal.app",
            "ApplicationProperties:ArchiveVersion": "2",
            "ApplicationProperties:CFBundleIdentifier": "com.google.example",
            "ApplicationProperties:CFBundleShortVersionString": "1.0",
            "ApplicationProperties:CFBundleVersion": "1.0",
            "Name": "app_minimal",
            "SchemeName": "app_minimal",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
