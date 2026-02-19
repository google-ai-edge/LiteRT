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

"""Starlark test rules for matching text in (non-bundle) rule outputs."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)

def _output_text_match_test_impl(ctx):
    """Implementation of the `output_text_match_test` rule."""
    target_under_test = ctx.attr.target_under_test

    path_suffixes = dicts.add(
        ctx.attr.files_match,
        ctx.attr.files_not_match,
    ).keys()

    # Map the path suffixes to files output by the target. If multiple outputs
    # match, fail the build.
    path_suffix_to_output = {}
    for path_suffix in path_suffixes:
        for output in target_under_test[DefaultInfo].files.to_list():
            if output.short_path.endswith(path_suffix):
                if path_suffix in path_suffix_to_output:
                    fail(("Target {} had multiple outputs whose paths end in " +
                          "'{}'; use additional path segments to distinguish " +
                          "them.").format(target_under_test.label, path_suffix))
                path_suffix_to_output[path_suffix] = output

    # If a path suffix did not match any of the outputs, fail.
    for path_suffix in path_suffixes:
        if path_suffix not in path_suffix_to_output:
            fail(("Target {} did not output a file whose path ends in " +
                  "'{}'.").format(target_under_test.label, path_suffix))

    # Generate a script that uses the regex matching assertions from
    # unittest.bash to verify the matches (or not-matches) in the outputs.
    unittest_bash_path = "test/unittest.bash"
    workspace = target_under_test.label.workspace_name
    if workspace != "":
        unittest_bash_path = paths.join("..", workspace, unittest_bash_path)
    generated_script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source {}".format(unittest_bash_path),
    ]
    for path_suffix, patterns in ctx.attr.files_match.items():
        for pattern in patterns:
            generated_script.append("assert_contains '{}' \"{}\"".format(
                pattern,
                path_suffix_to_output[path_suffix].short_path,
            ))

    for path_suffix, patterns in ctx.attr.files_not_match.items():
        for pattern in patterns:
            generated_script.append("assert_not_contains '{}' \"{}\"".format(
                pattern,
                path_suffix_to_output[path_suffix].short_path,
            ))

    output_script = ctx.actions.declare_file(
        "{}_test_script".format(ctx.label.name),
    )
    ctx.actions.write(
        output = output_script,
        content = "\n".join(generated_script),
        is_executable = True,
    )

    return [
        DefaultInfo(
            executable = output_script,
            runfiles = ctx.runfiles(
                files = (
                    path_suffix_to_output.values() +
                    ctx.attr._test_deps.files.to_list()
                ),
            ),
        ),
    ]

output_text_match_test = rule(
    attrs = {
        "files_match": attr.string_list_dict(
            mandatory = False,
            doc = """\
A dictionary where each key is the path suffix of a file output by the target
under test, and the corresponding value is a list of regular expressions that
are expected to be found somewhere in that file.
""",
        ),
        "files_not_match": attr.string_list_dict(
            mandatory = False,
            doc = """\
A dictionary where each key is the path suffix of a file output by the target
under test, and the corresponding value is a list of regular expressions that
are expected to **not** be found somewhere in that file.
""",
        ),
        "target_under_test": attr.label(
            mandatory = True,
            doc = "The target whose outputs are to be verified.",
        ),
        "_test_deps": attr.label(
            default = "//test:apple_verification_test_deps",
        ),
    },
    implementation = _output_text_match_test_impl,
    test = True,
)
