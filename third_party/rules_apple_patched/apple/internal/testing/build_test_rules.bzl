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

"""Rules for writing build tests for libraries that target Apple platforms."""

load(
    "//apple/internal:providers.bzl",
    "AppleBinaryInfo",
    "AppleDsymBundleInfo",
)
load(
    "//apple/internal:transition_support.bzl",
    "transition_support",
)

_PASSING_TEST_SCRIPT = """\
#!/bin/bash
exit 0
"""

# These providers mark major Apple targets that already contain transitions so
# there is no reason for a `PLATFORM_build_test` to wrap one of these, instead
# a plan `build_test` should be used.
_BLOCKED_PROVIDERS = [
    AppleBinaryInfo,
    AppleDsymBundleInfo,
]

def _apple_build_test_rule_impl(ctx):
    if ctx.attr.platform_type != ctx.attr._platform_type:
        fail((
            "The 'platform_type' attribute of '{}' is an implementation " +
            "detail and will be removed in the future; do not change it."
        ).format(ctx.attr._platform_type + "_build_test"))

    # TODO: b/293611241 - Check if the targets return any providers that
    # indicate that it would be better tested with a regular `build_test`
    # instead, and fail with a useful error message.
    targets = ctx.attr.targets
    for target in targets:
        for p in _BLOCKED_PROVIDERS:
            if p in target:
                fail((
                    "'{target_label}' builds a bundle and should just be " +
                    " wrapped with a 'build_test' and not '{rule_kind}'."
                ).format(
                    target_label = target.label,
                    rule_kind = ctx.attr._platform_type + "_build_test",
                ))

    transitive_files = [target[DefaultInfo].files for target in targets]

    # The test's executable is a vacuously passing script. We pass all of the
    # default outputs from the list of targets as the test's runfiles, so as
    # long as they all build successfully, the entire test will pass.
    ctx.actions.write(
        content = _PASSING_TEST_SCRIPT,
        output = ctx.outputs.executable,
        is_executable = True,
    )

    return [DefaultInfo(
        executable = ctx.outputs.executable,
        runfiles = ctx.runfiles(
            transitive_files = depset(transitive = transitive_files),
        ),
    )]

def apple_build_test_rule(doc, platform_type):
    """Creates and returns an Apple build test rule for the given platform.

    Args:
        doc: The documentation string for the rule.
        platform_type: The Apple platform for which the test should build its
            targets (`"ios"`, `"macos"`, `"tvos"`, `"visionos"`, or `"watchos"`).

    Returns:
        The created `rule`.
    """

    # TODO(b/161808913): Once resource processing actions have all been moved
    #  into the resource aspect (that is, they are processed at the library
    # level), apply the aspect to the targets and collect the processed
    # resource outputs so that the build test can verify that resources also
    # compile successfully; right now we just verify that the code in the
    # libraries compiles.
    return rule(
        attrs = {
            "minimum_os_version": attr.string(
                mandatory = True,
                doc = """\
A required string indicating the minimum OS version that will be used as the
deployment target when building the targets, represented as a dotted version
number (for example, `"9.0"`).
""",
            ),
            "targets": attr.label_list(
                allow_empty = False,
                cfg = transition_support.apple_platform_split_transition,
                doc = "The targets to check for successful build.",
            ),
            # This is a public attribute due to an implementation detail of
            # `apple_platform_split_transition`. The private attribute of the
            # same name is used in the implementation function to verify that
            # the user has not modified it.
            "platform_type": attr.string(default = platform_type),
            "_platform_type": attr.string(default = platform_type),
            "_allowlist_function_transition": attr.label(
                default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
            ),
        },
        doc = doc,
        implementation = _apple_build_test_rule_impl,
        test = True,
        cfg = transition_support.apple_rule_transition,
    )
