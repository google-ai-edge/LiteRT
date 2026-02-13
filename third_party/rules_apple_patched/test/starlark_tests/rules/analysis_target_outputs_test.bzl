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

"Starlark test for testing the outputs of analysis phase."

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
)
load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)
load(
    "//test/starlark_tests/rules:assertions.bzl",
    "assertions",
)

def _analysis_target_outputs_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)

    assertions.contains_files(
        env = env,
        expected_files = ctx.attr.expected_outputs,
        actual_files = target_under_test.files.to_list(),
    )

    return analysistest.end(env)

def make_analysis_target_outputs_test(config_settings = {}):
    """Returns a new `analysis_target_outputs_test`-like rule with custom configs.

    Args:
        config_settings: A dictionary of configuration settings and their values
            that should be applied during tests.

    Returns:
        A rule returned by `analysistest.make` that has the
        `analysis_target_outputs_test` interface and the given config settings.
    """
    return analysistest.make(
        _analysis_target_outputs_test_impl,
        config_settings = config_settings,
        attrs = {
            "expected_outputs": attr.string_list(
                doc = "The outputs that are expected.",
                default = [],
            ),
        },
    )

# Default analysis_target_actions_test.
analysis_target_outputs_test = make_analysis_target_outputs_test(
    config_settings = {
        "//command_line_option:compilation_mode": "opt",
    },
)

analysis_target_tree_artifacts_outputs_test = make_analysis_target_outputs_test(
    config_settings = {
        build_settings_labels.use_tree_artifacts_outputs: True,
    },
)
