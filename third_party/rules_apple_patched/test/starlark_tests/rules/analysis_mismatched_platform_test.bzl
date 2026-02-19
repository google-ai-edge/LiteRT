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

"""Starlark test testing failures when the incoming platform does not match the rule being built.
https://docs.bazel.build/versions/0.27.0/skylark/testing.html#failure-testing
"""

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)

def _analysis_incoming_watchos_platform_mismatch_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(
        env,
        """
ERROR: Unexpected resolved platform:
Expected Apple platform type of \"{expected}\", but that was not found in {platform}.
""".format(
            expected = ctx.attr.expected_platform_type,
            platform = Label("@build_bazel_apple_support//platforms:watchos_x86_64"),
        ),
    )
    return analysistest.end(env)

analysis_incoming_watchos_platform_mismatch_test = analysistest.make(
    _analysis_incoming_watchos_platform_mismatch_test_impl,
    expect_failure = True,
    attrs = {
        "expected_platform_type": attr.string(
            mandatory = True,
            doc = "Expected platform type for the target.",
        ),
    },
    config_settings = {
        "//command_line_option:incompatible_enable_apple_toolchain_resolution": True,
        "//command_line_option:platforms": [
            str(Label("@build_bazel_apple_support//platforms:watchos_x86_64")),
        ],
    },
)

def _analysis_incoming_ios_platform_mismatch_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(
        env,
        """
ERROR: Unexpected resolved platform:
Expected Apple platform type of \"{expected}\", but that was not found in {platform}.
""".format(
            expected = ctx.attr.expected_platform_type,
            platform = Label("@build_bazel_apple_support//platforms:ios_sim_arm64"),
        ),
    )
    return analysistest.end(env)

analysis_incoming_ios_platform_mismatch_test = analysistest.make(
    _analysis_incoming_ios_platform_mismatch_test_impl,
    expect_failure = True,
    attrs = {
        "expected_platform_type": attr.string(
            mandatory = True,
            doc = "Expected platform type for the target.",
        ),
    },
    config_settings = {
        "//command_line_option:incompatible_enable_apple_toolchain_resolution": True,
        "//command_line_option:platforms": [
            str(Label("@build_bazel_apple_support//platforms:ios_sim_arm64")),
        ],
    },
)
