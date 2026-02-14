# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""xctrunner Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

def xctrunner_test_suite(name):
    """Test suite for xctrunner rule.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Verify xctrunner bundles required files for device.
    archive_contents_test(
        name = "{}_contains_xctrunner_files_device".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:ui_test_xctrunner_app",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Plugins/ui_test.xctest",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "CFBundleExecutable": "ui_test_xctrunner_app",
            "CFBundleIdentifier": "com.apple.test.ui_test_xctrunner_app",
            "CFBundleName": "ui_test_xctrunner_app",
            "DTPlatformName": "iphoneos",
        },
        tags = [name],
    )

    # Verify xctrunner bundles multiple targets for device.
    archive_contents_test(
        name = "{}_contains_multiple_targets_device".format(name),
        build_type = "device",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:ui_test_xctrunner_app_multiple_targets",
        contains = [
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Plugins/ui_test.xctest",
            "$BUNDLE_ROOT/Plugins/ui_test_with_fmwk.xctest",
        ],
        plist_test_file = "$BUNDLE_ROOT/Info.plist",
        plist_test_values = {
            "CFBundleExecutable": "ui_test_xctrunner_app_multiple_targets",
            "CFBundleIdentifier": "com.apple.test.ui_test_xctrunner_app_multiple_targets",
            "CFBundleName": "ui_test_xctrunner_app_multiple_targets",
            "DTPlatformName": "iphoneos",
        },
        tags = [name],
    )
