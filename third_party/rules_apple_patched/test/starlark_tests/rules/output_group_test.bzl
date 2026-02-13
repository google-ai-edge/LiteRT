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

"""Starlark test rules for output groups."""

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)

def _output_group_test_impl(ctx):
    """Implementation of the output_group_test rule."""
    env = analysistest.begin(ctx)
    target_under_test = ctx.attr.target_under_test
    expected_groups = ctx.attr.expected_output_groups

    for expected_group in expected_groups:
        asserts.true(
            env,
            expected_group in target_under_test[OutputGroupInfo],
            msg = "Expected output group not found\n\n\"{0}\"".format(
                expected_group,
            ),
        )

    return analysistest.end(env)

output_group_test = analysistest.make(
    _output_group_test_impl,
    attrs = {
        "expected_output_groups": attr.string_list(
            mandatory = True,
            doc = """List of output groups that should be present in the target.""",
        ),
    },
)
