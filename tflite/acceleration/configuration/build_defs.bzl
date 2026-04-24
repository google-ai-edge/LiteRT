# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build macros for checking ABI compatibility."""

load("@flatbuffers//:build_defs.bzl", "flatc_path")
load("@rules_shell//shell:sh_test.bzl", "sh_test")

def flatbuffer_schema_compat_test(name, ref_schema, schema):
    """Generates a test for schema binary compatibility.

    Generates a test that the specified schema file is binary backwards
    compatible with a reference schema (e.g. a previous version of the
    schema).

    Note: currently this build macro requires that the schema be a single
    fully self-contained .fbs file; it does not yet support includes.
    """

    native.genrule(
        name = name + "_gen",
        srcs = [ref_schema, schema],
        outs = [name + "_test.sh"],
        tools = [flatc_path],
        cmd = ("echo $(rootpath {}) --conform $(rootpath {}) $(rootpath {}) > $@"
            .format(flatc_path, ref_schema, schema)),
    )
    sh_test(
        name = name,
        srcs = [name + "_test.sh"],
        data = [flatc_path, ref_schema, schema],
    )

def _flatbuffer_schema_from_proto_impl(ctx):
    args = ctx.actions.args()
    args.add("generate_schema")
    args.add("--flatc")
    args.add(ctx.executable._flatc)
    args.add("--src")
    args.add(ctx.file.src)
    args.add("--out")
    args.add(ctx.outputs.out)
    if ctx.attr.proto_name:
        args.add("--proto_name")
        args.add(ctx.attr.proto_name)

    ctx.actions.run(
        executable = ctx.executable._schema_tool,
        arguments = [args],
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        tools = [
            ctx.attr._schema_tool[DefaultInfo].files_to_run,
            ctx.attr._flatc[DefaultInfo].files_to_run,
        ],
        mnemonic = "FlatbufferSchemaFromProto",
        progress_message = "Generating FlatBuffer schema %{output}",
        use_default_shell_env = False,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

flatbuffer_schema_from_proto = rule(
    implementation = _flatbuffer_schema_from_proto_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "proto_name": attr.string(),
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "_flatc": attr.label(
            default = Label("@flatbuffers//:flatc"),
            executable = True,
            cfg = "exec",
        ),
        "_schema_tool": attr.label(
            default = Label("//tflite/acceleration/configuration:configuration_schema_tool"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def _flatbuffer_schema_contents_header_impl(ctx):
    args = ctx.actions.args()
    args.add("generate_header")
    args.add("--src")
    args.add(ctx.file.src)
    args.add("--out")
    args.add(ctx.outputs.out)
    args.add("--variable_name")
    args.add(ctx.attr.variable_name)

    ctx.actions.run(
        executable = ctx.executable._schema_tool,
        arguments = [args],
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        tools = [ctx.attr._schema_tool[DefaultInfo].files_to_run],
        mnemonic = "FlatbufferSchemaContentsHeader",
        progress_message = "Generating FlatBuffer schema contents header %{output}",
        use_default_shell_env = False,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

flatbuffer_schema_contents_header = rule(
    implementation = _flatbuffer_schema_contents_header_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "variable_name": attr.string(default = "configuration_fbs_contents"),
        "_schema_tool": attr.label(
            default = Label("//tflite/acceleration/configuration:configuration_schema_tool"),
            executable = True,
            cfg = "exec",
        ),
    },
)
