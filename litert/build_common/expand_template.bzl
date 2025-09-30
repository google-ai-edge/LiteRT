# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Template expansion that supports bazel variable resolution.."""

def _quote(s):
    return '"%s"' % s

def _extract_binary(target):
    # cc_test case
    if RunEnvironmentInfo in target:
        environment = target[RunEnvironmentInfo].environment
        cc_test_binary = environment.get("CC_TEST_BINARY", None)
        if cc_test_binary:
            return cc_test_binary

    # cc_binary case
    file_to_run = target.files_to_run.executable
    if file_to_run:
        return file_to_run.short_path

    fail("Could not locate an executable file provided by the target, excpected cc_test or cc_binary.")

def _fmt_location(mode, files):
    if mode != "array" and len(mode) != 1:
        fail("Unknown multi_output_behavior: %s" % mode)

    if len(files) == 0:
        fail("No files found for location expansion.")

    if len(files) == 1:
        return _quote(files[0].short_path)

    if mode == "array":
        return "({})".format(" ".join([_quote(f.short_path) for f in files]))

    return _quote(mode.join([f.short_path for f in files]))

def _expand_template_impl(ctx):
    forwarded_runfiles = []
    subs = ctx.actions.template_dict()

    # Basic string substitutions.
    for k, v in ctx.attr.subs.items():
        subs.add(k, v)

    # Data files location substitutions.
    for k, v in ctx.attr.data_subs.items():
        files = v.data_runfiles.files.to_list()
        if not files:
            files = [f for f in v.files if f.is_source]
        forwarded_runfiles.extend(files)
        subs.add(k, _fmt_location(ctx.attr.multi_output_delim, files))

    # Binary files location substitutions.
    for k, v in ctx.attr.bin_subs.items():
        forwarded_runfiles.extend(v.default_runfiles.files.to_list())
        binary = _extract_binary(v)
        subs.add(k, _quote(binary))

    runfiles = ctx.runfiles(files = forwarded_runfiles)

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        computed_substitutions = subs,
        is_executable = ctx.attr.executable,
    )

    return [DefaultInfo(runfiles = runfiles)]

expand_template = rule(
    implementation = _expand_template_impl,
    doc = """
    Template expansion with location and executable support. Will also forward any runfiles 
    (included transitively) substituted into the template.
    """,
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The template file to expand.",
        ),
        "subs": attr.string_dict(
            mandatory = True,
            doc = "A dictionary mapping strings to their substitutions.",
        ),
        "data_subs": attr.string_keyed_label_dict(
            default = {},
            doc = """
            A dictionary mapping strings to substituted labels. Labels we be interpreted as data
            dependencies, so its resolved value will reflect the location of all
            (including transitive) data runfiles from said label.
            """,
            allow_files = True,
        ),
        "bin_subs": attr.string_keyed_label_dict(
            default = {},
            doc = """
            A dictionary mapping strings to substituted labels. Labels will be interpreted as
            binary dependencies, so its resolved value will reflect the location of the sole binary
            executable.
            """,
            providers = [RunEnvironmentInfo],
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The destination of the expanded file.",
        ),
        "executable": attr.bool(
            default = False,
            doc = "Whether to make the output executable.",
        ),
        "multi_output_delim": attr.string(
            default = "array",
            doc = """
            How to format the string when resolving the location of a target with multiple outputs. 
            \"array\" will be a shell array, anything will serve as a delimiter.
            """,
        ),
    },
)
