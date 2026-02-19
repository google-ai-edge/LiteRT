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

"""macos_extension Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
    "entry_point_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)

def macos_extension_test_suite(name):
    """Test suite for macos_extension.

    Args:
      name: the base name to be used in things created by this macro
    """
    entry_point_test(
        name = "{}_entry_point_nsextensionmain_test".format(name),
        build_type = "simulator",
        entry_point = "_NSExtensionMain",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:ext",
        tags = [name],
    )

    archive_contents_test(
        name = "{}_correct_rpath_header_value_test".format(name),
        build_type = "device",
        binary_test_file = "$CONTENT_ROOT/MacOS/ext",
        macho_load_commands_contain = [
            "path @executable_path/../Frameworks (offset 12)",
            "path @executable_path/../../../../Frameworks (offset 12)",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/macos:ext",
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_capability_set_derived_bundle_id_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:ext_with_capability_set_derived_bundle_id",
        expected_values = {
            "CFBundleIdentifier": "com.bazel.app.example.ext-with-capability-set-derived-bundle-id",
        },
        tags = [name],
    )

    # Test that an ExtensionKit extension is bundled in Extensions and not PlugIns.
    archive_contents_test(
        name = "{}_extensionkit_bundling_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/macos:app_with_extensionkit_ext",
        contains = ["$BUNDLE_ROOT/Contents/Extensions/extensionkit_ext.appex/Contents/MacOS/extensionkit_ext"],
        not_contains = ["$BUNDLE_ROOT/Contents/PlugIns/extensionkit_ext.appex/Contents/MacOS/extensionkit_ext"],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
