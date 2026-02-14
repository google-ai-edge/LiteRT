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

"""Apple capability rule Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)

visibility("private")

def apple_capability_test_suite(name):
    """Test suite for Apple capability rules.

    Args:
      name: the base name to be used in things created by this macro
    """
    analysis_failure_message_test(
        name = "{}_just_dot_base_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple/capabilities:just_dot_base_bundle_id",
        expected_error = "Empty segment in bundle_id: \".\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_leading_dot_base_bundle_id".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple/capabilities:leading_dot_base_bundle_id",
        expected_error = "Empty segment in bundle_id: \".com.bazel.app\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_trailing_dot_base_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple/capabilities:trailing_dot_base_bundle_id",
        expected_error = "Empty segment in bundle_id: \"com.bazel.app.\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_double_dot_base_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple/capabilities:double_dot_base_bundle_id",
        expected_error = "Empty segment in bundle_id: \"com..bazel.app\"",
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_invalid_character_bundle_id_fail_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple/capabilities:invalid_character_bundle_id",
        expected_error = "Invalid character(s) in bundle_id: \"com#bazel.app\"",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
