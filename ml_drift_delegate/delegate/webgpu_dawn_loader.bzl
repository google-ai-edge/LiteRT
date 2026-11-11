# Copyright 2026 Google LLC.
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

"""Starlark rule to generate DawnProcTable dynamic symbol population helper."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _generate_webgpu_dawn_loader_inc_impl(ctx):
    proc_table_h = None
    for hdr in ctx.attr.dawn_headers[CcInfo].compilation_context.headers.to_list():
        if hdr.path.endswith("/dawn/dawn_proc_table.h") or hdr.short_path.endswith("dawn/dawn_proc_table.h"):
            proc_table_h = hdr
            break

    if proc_table_h == None:
        fail("Failed to find dawn/dawn_proc_table.h in dawn_headers")

    args = ctx.actions.args()
    args.add("--proc_table_header", proc_table_h)
    args.add("--output", ctx.outputs.out)

    ctx.actions.run(
        inputs = [proc_table_h],
        outputs = [ctx.outputs.out],
        executable = ctx.executable._generator,
        arguments = [args],
        mnemonic = "GenWebGpuDawnLoader",
        progress_message = "Generating WebGPU DawnProcTable loader %{output}",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_webgpu_dawn_loader_inc = rule(
    implementation = _generate_webgpu_dawn_loader_inc_impl,
    attrs = {
        "dawn_headers": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
        "out": attr.output(mandatory = True),
        "_generator": attr.label(
            default = Label("//ml_drift_delegate/delegate:webgpu_dawn_loader_gen"),
            executable = True,
            cfg = "exec",
        ),
    },
)
