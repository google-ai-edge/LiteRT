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

"""Starlark analysis test inspecting target under test actions."""

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "unittest",
)
load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)

_TARGET_CONTAINS_ACTION_WITH_ARGV_FAIL_MSG = """
Expected argv could not be found on actual action argv list for target mnemonic '{target_mnemonic}'.
Target: {target}
Expected action argv: {expected_argv}
Actual action argv: {actual_argv}
"""

_TARGET_CONTAINS_ACTION_WITH_ENV_FAIL_MSG = """
Expected env could not be found on actual action env list for target mnemonic '{target_mnemonic}'.
Target: {target}
Expected action env: {expected_env}
Actual action env: {actual_env}
"""

_TARGET_CONTAINS_NOT_EXPECTED_MNEMONIC = """
Expected target to not contain an action with mnemonic '{target_mnemonic}', but it did.
Target: {target}
Mnemonic: {target_mnemonic}
Action argv: {action_argv}
"""

def _analysis_target_actions_test_impl(ctx):
    """Implementation of analysis_target_actions_test."""
    env = analysistest.begin(ctx)
    target_mnemonic = ctx.attr.target_mnemonic
    target_actions = analysistest.target_actions(env)
    target_under_test = analysistest.target_under_test(env)

    target_mnemonic_actions = [a for a in target_actions if a.mnemonic == target_mnemonic]

    if not target_mnemonic_actions:
        unittest.fail(env, "Could not find any matching actions for mnemonic: %s" % target_mnemonic)
        return analysistest.end(env)

    for expected_argv in ctx.attr.expected_argv:
        # Concatenate the arguments into a single string so that we can easily look
        # for subsequences of arguments. Note that we append an extra space to the
        # end and look for arguments followed by a trailing space so that having
        # `-foo` in the expected list doesn't match `-foobar`, for example.
        target_mnemonic_actions_argv = [
            " ".join(a.argv) + " "
            for a in target_mnemonic_actions
            if a.argv != None
        ]

        matched_expected_argv = False
        for action_argv in target_mnemonic_actions_argv:
            if expected_argv + " " in action_argv:
                matched_expected_argv = True
                break

        if not matched_expected_argv:
            unittest.fail(
                env,
                _TARGET_CONTAINS_ACTION_WITH_ARGV_FAIL_MSG.format(
                    target_mnemonic = target_mnemonic,
                    target = target_under_test,
                    expected_argv = expected_argv,
                    actual_argv = target_mnemonic_actions_argv,
                ),
            )
            return analysistest.end(env)

    for expected_env_key, expected_env_value in ctx.attr.expected_env.items():
        target_mnemonic_actions_env = [a.env for a in target_mnemonic_actions]

        matched_expected_env = False
        for action_env in target_mnemonic_actions_env:
            if expected_env_key not in action_env:
                continue
            if action_env[expected_env_key] == expected_env_value:
                matched_expected_env = True
                break
        if not matched_expected_env:
            unittest.fail(
                env,
                _TARGET_CONTAINS_ACTION_WITH_ENV_FAIL_MSG.format(
                    target_mnemonic = target_mnemonic,
                    target = target_under_test,
                    expected_env = {expected_env_key: expected_env_value},
                    actual_env = target_mnemonic_actions_env,
                ),
            )
            return analysistest.end(env)

    for not_expected_mnemonic in ctx.attr.not_expected_mnemonic:
        actual_mnemonics = {
            action.mnemonic: action
            for action in target_actions
            if action.mnemonic == not_expected_mnemonic
        }
        if not_expected_mnemonic in actual_mnemonics:
            action = actual_mnemonics[not_expected_mnemonic]
            unittest.fail(
                env,
                _TARGET_CONTAINS_NOT_EXPECTED_MNEMONIC.format(
                    target_mnemonic = not_expected_mnemonic,
                    target = target_under_test,
                    action_argv = getattr(action, "argv"),
                ),
            )
            return analysistest.end(env)

    return analysistest.end(env)

def make_analysis_target_actions_test(config_settings = {}):
    """Returns a new `analysis_target_actions_test`-like rule with custom configs.

    Args:
        config_settings: A dictionary of configuration settings and their values
            that should be applied during tests.

    Returns:
        A rule returned by `analysistest.make` that has the
        `analysis_target_actions_test` interface and the given config settings.
    """
    return analysistest.make(
        _analysis_target_actions_test_impl,
        attrs = {
            "target_mnemonic": attr.string(
                mandatory = True,
                doc = """
The mnemonic of the action to be inspected on the target under test.
This will also assert at least one action exists with the given mnemonic.
""",
            ),
            "expected_argv": attr.string_list(
                doc = """
A list of strings representing substrings expected to appear in the action
command line, after concatenating all command line arguments into a single
space-delimited string.""",
            ),
            "expected_env": attr.string_dict(
                doc = """
A string dictionary representing expected environment values that should be
present in the action environment values.""",
            ),
            "not_expected_mnemonic": attr.string_list(
                doc = """
List of action mnemonics not expected to be found on the target under test.""",
            ),
        },
        config_settings = config_settings,
    )

# Default analysis_target_actions_test without cfg.
analysis_target_actions_test = make_analysis_target_actions_test()

# The following test rules are used in more than one test suite and thus they are defined here.
analysis_contains_xcframework_processor_action_test = make_analysis_target_actions_test(
    config_settings = {
        build_settings_labels.use_tree_artifacts_outputs: True,
    },
)
analysis_target_actions_tree_artifacts_outputs_test = make_analysis_target_actions_test(
    config_settings = {
        build_settings_labels.use_tree_artifacts_outputs: True,
    },
)
