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

"""Starlark analysis test assertions for tests."""

load(
    "@bazel_skylib//lib:new_sets.bzl",
    "sets",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)

visibility("//test/starlark_tests/...")

def _contains_files(env, expected_files, actual_files):
    target_under_test = analysistest.target_under_test(env)
    expected_set = sets.make(expected_files)

    all_outputs = sets.make([
        paths.relativize(
            file.short_path,
            target_under_test.label.package,
        )
        for file in actual_files
    ])

    # Test that the expected outputs are contained within actual outputs
    asserts.set_equals(
        env,
        expected_set,
        sets.intersection(all_outputs, expected_set),
        "{} not contained in {}".format(sets.to_list(expected_set), sets.to_list(all_outputs)),
    )

assertions = struct(
    contains_files = _contains_files,
)
