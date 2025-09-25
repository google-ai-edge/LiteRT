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
    subs = ctx.actions.template_dict()

    for k, v in ctx.attr.subs.items():
        subs.add(k, v)

    forwarded_runfiles = []
    for k, v in ctx.attr.run_subs.items():
        files = v.default_runfiles.files.to_list()
        forwarded_runfiles.extend(files)
        subs.add(k, _fmt_location(ctx.attr.multi_output_delim, files))

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
    substituted into the template.
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
        "run_subs": attr.string_keyed_label_dict(
            default = {},
            doc = "A dictionary mapping strings to substituted labels. These will be resolved to their runfiles (rlocationpath) locations.",
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
            doc = "How to format the string when resolving the location of a target with multiple outputs. \"array\" will be a shell array, anything will serve as a delimiter.",
        ),
    },
)
