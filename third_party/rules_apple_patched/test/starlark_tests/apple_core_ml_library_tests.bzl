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

"""apple_core_ml_library Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

visibility("private")

def apple_core_ml_library_test_suite(name):
    """Test suite for apple_core_ml_library.

    Args:
      name: the base name to be used in things created by this macro
    """

    archive_contents_test(
        name = "{}_contains_core_ml_model".format(name),
        build_type = "simulator",
        contains = ["$BUNDLE_ROOT/sample.mlmodelc/"],
        target_under_test = "//test/starlark_tests/targets_under_test/apple:app_with_core_ml_library",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
