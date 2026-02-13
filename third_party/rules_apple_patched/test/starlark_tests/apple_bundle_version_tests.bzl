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

"""apple_bundle_version Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)

def apple_bundle_version_test_suite(name):
    """Test suite for apple_bundle_version.

    Args:
      name: the base name to be used in things created by this macro
    """

    # Tests that manual version numbers work correctly.
    infoplist_contents_test(
        name = "{}_manual_version".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:manual_1_2_build_1_2_3_bundle",
        expected_values = {
            "CFBundleVersion": "1.2.3",
            "CFBundleShortVersionString": "1.2",
        },
        tags = [name],
    )

    # Tests that short_version_string defaults to the same value as build_version.
    infoplist_contents_test(
        name = "{}_short_version_defaults_to_build_version".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:only_build_version_1_2_3_bundle",
        expected_values = {
            "CFBundleVersion": "1.2.3",
            "CFBundleShortVersionString": "1.2.3",
        },
        tags = [name],
    )

    # Test that the fallback_build_label is used when --embed_label is not passed on the build.
    infoplist_contents_test(
        name = "{}_build_label_substitution_from_fallback_label".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:build_label_substitution_from_fallback_label_bundle",
        expected_values = {
            "CFBundleVersion": "99.99.99",
            "CFBundleShortVersionString": "99.99",
        },
        tags = [name],
    )

    # Tests that short_version_string defaults to the same value as build_version when using
    # substitution.
    infoplist_contents_test(
        name = "{}_short_version_string_defaults_to_build_version_substitution".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:short_version_string_defaults_to_build_version_substitution_bundle",
        expected_values = {
            "CFBundleVersion": "1.2.3",
            "CFBundleShortVersionString": "1.2.3",
        },
        tags = [name],
    )

    # Tests that a build_label_pattern that contains placeholders not found in capture_groups
    # causes analysis to fail.
    analysis_failure_message_test(
        name = "{}_pattern_referencing_missing_capture_groups_fail".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:pattern_referencing_missing_capture_groups_fail",
        expected_error = "Some groups were not defined in capture_groups",
        tags = [name],
    )

    # Tests that the analysis fails if build_label_pattern is provided but capture_groups
    # is not.
    analysis_failure_message_test(
        name = "{}_build_label_pattern_requires_capture_groups_fail".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:build_label_pattern_requires_capture_groups_fail",
        expected_error = "If either build_label_pattern or capture_groups is provided, then both must be provided.",
        tags = [name],
    )

    # Tests that the analysis fails if capture_groups is provided but build_label_pattern
    # is not.
    analysis_failure_message_test(
        name = "{}_capture_groups_requires_build_label_pattern_fail".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:capture_groups_requires_build_label_pattern_fail",
        expected_error = "If either build_label_pattern or capture_groups is provided, then both must be provided.",
        tags = [name],
    )

    # Tests that the analysis fails if fallback_build_label is provided but build_label_pattern
    # is not.
    analysis_failure_message_test(
        name = "{}_fallback_build_label_requires_build_label_pattern_fail".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:fallback_build_label_requires_build_label_pattern_fail",
        expected_error = "If fallback_build_label is provided, then build_label_pattern and capture_groups must be provided.",
        tags = [name],
    )

    # Test that substitution does not occur if there is a build label pattern but --embed_label
    # is not specified on the command line. (This supports local builds).
    infoplist_contents_test(
        name = "{}_no_substitution_if_build_label_not_present".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:no_substitution_if_build_label_not_present_bundle",
        expected_values = {
            "CFBundleVersion": "1.0",
            "CFBundleShortVersionString": "1.0",
        },
        tags = [name],
    )

    # Test that the presence of a build label pattern does not short circuit the use of a literal
    # version string, even if no --embed_label argument is provided.
    infoplist_contents_test(
        name = "{}_build_label_pattern_does_not_short_circuit_literal".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:build_label_pattern_does_not_short_circuit_literal_bundle",
        expected_values = {
            "CFBundleVersion": "1.2.3",
            "CFBundleShortVersionString": "1.2",
        },
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
