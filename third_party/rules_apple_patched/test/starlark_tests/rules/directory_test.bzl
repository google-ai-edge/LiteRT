# Copyright 2025 The Bazel Authors. All rights reserved.
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

"""Rules for testing the contents of directory artifact outputs.

Since the contents of these directories are not known at analysis time, we need
to spawn a shell script that lists their contents.
"""

load(
    "//test/starlark_tests/rules:apple_verification_test.bzl",
    "apple_verification_transition",
)

visibility([
    "//test/starlark_tests/...",
])

def _directory_test_impl(ctx):
    target_under_test = ctx.attr.target_under_test[0]
    path_suffixes = ctx.attr.expected_directories.keys()

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
            fail(("Target {} did not output a directory whose path ends in " +
                  "'{}'.").format(target_under_test.label, path_suffix))

    # Generate a script that verifies the existence of each expected file.
    generated_script = [
        "#!/usr/bin/env bash",
        "function check_file() {",
        "  if [[ -f \"$1/$2\" ]]; then",
        "    return 0",
        "  else",
        "    echo \"ERROR: Expected file '$2' did not exist in output " +
        "directory '$1'\"",
        "    return 1",
        "  fi",
        "}",
        "failed=0",
    ]
    for path_suffix, files in ctx.attr.expected_directories.items():
        dir_path = path_suffix_to_output[path_suffix].short_path
        for file in files:
            generated_script.append(
                "check_file \"{dir}\" \"{file}\" || failed=1".format(
                    dir = dir_path,
                    file = file,
                ),
            )
        generated_script.append("echo")

    generated_script.append("exit ${failed}")

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
            runfiles = ctx.runfiles(files = path_suffix_to_output.values()),
        ),
    ]

directory_test = rule(
    attrs = {
        "apple_generate_dsym": attr.bool(
            default = False,
            doc = """
If true, generates .dSYM debug symbol bundles for the target(s) under test.
""",
        ),
        "build_settings": attr.string_dict(
            mandatory = False,
            doc = "Build settings for target under test.",
        ),
        "build_type": attr.string(
            default = "simulator",
            doc = """
Type of build for the target under test. Possible values are `simulator` or `device`.
Defaults to `simulator`.
""",
            values = ["simulator", "device"],
        ),
        "compilation_mode": attr.string(
            default = "fastbuild",
            doc = """
Possible values are `fastbuild`, `dbg` or `opt`. Defaults to `fastbuild`.
https://docs.bazel.build/versions/master/user-manual.html#flag--compilation_mode
""",
            values = ["fastbuild", "opt", "dbg"],
        ),
        "expected_directories": attr.string_list_dict(
            mandatory = True,
            doc = """\
A dictionary where each key is the path suffix of a directory (tree artifact)
output by the target under test, and the corresponding value is a list of files
expected to exist in that directory (expressed as paths relative to the key).
""",
        ),
        "target_under_test": attr.label(
            cfg = apple_verification_transition,
            doc = "The target whose outputs are to be verified.",
            mandatory = True,
        ),
    },
    implementation = _directory_test_impl,
    test = True,
)
