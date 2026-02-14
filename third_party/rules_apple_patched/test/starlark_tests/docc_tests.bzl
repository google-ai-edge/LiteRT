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

"""docc Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

def docc_test_suite(name):
    """Test suite for docc rules.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Verify doccarchive bundle is created for Swift iOS app.
    archive_contents_test(
        name = "{}_contains_doccarchive_when_ios_swift_app".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:app_with_swift_dep.doccarchive",
        contains = [
            "$BUNDLE_ROOT/index.html",
        ],
        text_file_not_contains = [
            "$BUNDLE_ROOT/documentation/basicframework/readme/index.html",  # only included with a .docc bundle in data
        ],
        text_test_file = "$BUNDLE_ROOT/metadata.json",
        text_test_values = [
            "\"bundleDisplayName\":\"app_with_swift_dep\"",
            "\"bundleID\":\"com.google.example\"",
            "\"major\":0",
            "\"minor\":1",
            "\"patch\":0",
        ],
        tags = [name],
    )

    # Verify doccarchive bundle is created for Swift iOS framework which includes a .docc bundle.
    archive_contents_test(
        name = "{}_contains_doccarchive_with_docc_bundle_when_ios_framework".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:basic_framework_with_docc_bundle.doccarchive",
        contains = [
            "$BUNDLE_ROOT/index.html",
            "$BUNDLE_ROOT/documentation/basicframework/readme/index.html",
        ],
        text_test_file = "$BUNDLE_ROOT/metadata.json",
        text_test_values = [
            "\"bundleDisplayName\":\"BasicFramework\"",
            "\"bundleID\":\"com.google.example.framework\"",
            "\"major\":0",
            "\"minor\":1",
            "\"patch\":0",
        ],
        tags = [name],
    )

    # Verify doccarchive bundle is created for an ObjC library which defines data and has a dependency on a Swift lib.
    archive_contents_test(
        name = "{}_contains_doccarchive_when_objc_library_with_swift_dep".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:basic_objc_lib_with_data_and_docc_bundle_dependency.doccarchive",
        contains = [
            "$BUNDLE_ROOT/index.html",
            "$BUNDLE_ROOT/documentation/basiclib/readme/index.html",
        ],
        text_test_file = "$BUNDLE_ROOT/metadata.json",
        text_test_values = [
            "\"bundleDisplayName\":\"BasicLib\"",
            "\"bundleID\":\"com.google.example.objc.lib\"",
            "\"major\":0",
            "\"minor\":1",
            "\"patch\":0",
        ],
        tags = [name],
    )

    # Verify hosting_base_path support.
    archive_contents_test(
        name = "{}_contains_doccarchive_with_docc_bundle_when_ios_framework_with_hosting_base_path".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:basic_framework_with_docc_bundle_custom_base_path.doccarchive",
        contains = [
            "$BUNDLE_ROOT/index.html",
            "$BUNDLE_ROOT/documentation/basicframework/readme/index.html",
        ],
        text_test_file = "$BUNDLE_ROOT/index.html",
        text_test_values = [
            "<script defer=\"defer\" src=\"/custom/base/path/js/",
            "<link href=\"/custom/base/path/css/",
        ],
        tags = [name],
    )

    # Verifying multiple symbol graph conversion via transitive dependencies.
    archive_contents_test(
        name = "{}_contains_doccarchive_with_transitive_dependencies".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/ios:basic_framework_with_transitive_dependency.doccarchive",
        contains = [
            "$BUNDLE_ROOT/index.html",
            "$BUNDLE_ROOT/documentation/transitivedependencytest/index.html",
        ],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
