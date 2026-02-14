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

"""Rules for testing the contents of action command lines."""

load("@bazel_skylib//lib:collections.bzl", "collections")
load("@bazel_skylib//lib:unittest.bzl", "analysistest", "unittest")

def _action_command_line_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)

    # Find the desired action and verify that there is exactly one.
    actions = analysistest.target_actions(env)
    mnemonic = ctx.attr.mnemonic
    matching_actions = [
        action
        for action in actions
        if action.mnemonic == mnemonic
    ]
    if not matching_actions:
        actual_mnemonics = collections.uniq(
            [action.mnemonic for action in actions],
        )
        unittest.fail(
            env,
            ("Target '{}' registered no actions with the mnemonic '{}' " +
             "(it had {}).").format(
                str(target_under_test.label),
                mnemonic,
                actual_mnemonics,
            ),
        )
        return analysistest.end(env)
    if len(matching_actions) != 1:
        # This is a hack to avoid CppLink meaning binary linking and static
        # library archiving https://github.com/bazelbuild/bazel/pull/15060
        if mnemonic == "CppLink" and len(matching_actions) == 2:
            matching_actions = [
                action
                for action in matching_actions
                if action.argv[0] not in [
                    "/usr/bin/ar",
                    "external/local_config_cc/libtool",
                ]
            ]

        # For multi-arch binaries assume the link actions are the same
        if mnemonic == "ObjcLink":
            matching_actions = [matching_actions[0]]

        if len(matching_actions) != 1:
            unittest.fail(
                env,
                ("Expected exactly one action with the mnemonic '{}', " +
                 "but found {}.").format(
                    mnemonic,
                    len(matching_actions),
                ),
            )
            return analysistest.end(env)

    action = matching_actions[0]
    message_prefix = "In {} action for target '{}', ".format(
        mnemonic,
        str(target_under_test.label),
    )

    # Concatenate the arguments into a single string so that we can easily look
    # for subsequences of arguments. Note that we append an extra space to the
    # end and look for arguments followed by a trailing space so that having
    # `-foo` in the expected list doesn't match `-foobar`, for example.
    concatenated_args = " ".join(action.argv) + " "
    for expected in ctx.attr.expected_argv:
        if expected + " " not in concatenated_args and expected + "=" not in concatenated_args:
            unittest.fail(
                env,
                "{}expected argv to contain '{}', but it did not: {}".format(
                    message_prefix,
                    expected,
                    concatenated_args,
                ),
            )
    for not_expected in ctx.attr.not_expected_argv:
        if not_expected + " " in concatenated_args or not_expected + "=" in concatenated_args:
            unittest.fail(
                env,
                "{}expected argv to not contain '{}', but it did: {}".format(
                    message_prefix,
                    not_expected,
                    concatenated_args,
                ),
            )

    return analysistest.end(env)

def make_action_command_line_test_rule(config_settings = {}):
    """Returns a new `action_command_line_test`-like rule with custom configs.

    Args:
        config_settings: A dictionary of configuration settings and their values
            that should be applied during tests.

    Returns:
        A rule returned by `analysistest.make` that has the
        `action_command_line_test` interface and the given config settings.
    """
    return analysistest.make(
        _action_command_line_test_impl,
        attrs = {
            "expected_argv": attr.string_list(
                mandatory = False,
                doc = """\
A list of strings representing substrings expected to appear in the action
command line, after concatenating all command line arguments into a single
space-delimited string.
""",
            ),
            "not_expected_argv": attr.string_list(
                mandatory = False,
                doc = """\
A list of strings representing substrings expected not to appear in the action
command line, after concatenating all command line arguments into a single
space-delimited string.
""",
            ),
            "mnemonic": attr.string(
                mandatory = True,
                doc = """\
The mnemonic of the action to be inspected on the target under test. It is
expected that there will be exactly one of these.
""",
            ),
        },
        config_settings = config_settings,
    )

# A default instantiation of the rule when no custom config settings are needed.
action_command_line_test = make_action_command_line_test_rule()
