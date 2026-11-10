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
"""A `platform_data` variant that also builds the target with YNNPACK enabled.

This is equivalent to `platform_data` from
//third_party/bazel_rules/rules_platform/platform_data:defs.bzl, except that the
transition also sets `//litert/build_common:enable_ynnpack`
(the Starlark-settable equivalent of `--define=litert_enable_ynnpack=true`).

Use this instead of a custom `platform(flags = ["--define=litert_enable_ynnpack=true"])`
when cross-compiling for macOS / Windows: the Rust proc-macro rules select the exec
Rust toolchain based on the platform label (i.e. the canonical Apple / Windows
platform labels), so a custom platform (even with a canonical parent) makes Rust
proc-macros get built with the wrong toolchain and breaks the build.
See http://b/479222736#comment12.
"""

_ENABLE_YNNPACK_SETTING = "//litert/build_common:enable_ynnpack"

def _ynnpack_platform_transition_impl(_settings, attr):
    return {
        "//command_line_option:platforms": str(attr.platform),
        _ENABLE_YNNPACK_SETTING: True,
    }

_ynnpack_platform_transition = transition(
    implementation = _ynnpack_platform_transition_impl,
    inputs = [],
    outputs = [
        "//command_line_option:platforms",
        _ENABLE_YNNPACK_SETTING,
    ],
)

def _ynnpack_platform_data_impl(ctx):
    target = ctx.attr.target
    default_info = target[0][DefaultInfo]
    original_executable = default_info.files_to_run.executable

    new_executable = ctx.actions.declare_file(ctx.attr.name)
    ctx.actions.symlink(
        output = new_executable,
        target_file = original_executable,
        is_executable = True,
    )

    files = depset(direct = [new_executable], transitive = [default_info.files])
    runfiles = default_info.default_runfiles.merge(ctx.runfiles([new_executable]))

    return [
        DefaultInfo(
            files = files,
            runfiles = runfiles,
            executable = new_executable,
        ),
    ]

ynnpack_platform_data = rule(
    implementation = _ynnpack_platform_data_impl,
    doc = """Builds `target` for `platform` with YNNPACK enabled.

`platform` should be a canonical platform label, not a custom platform with
`flags`, so that Rust proc-macros are built correctly.""",
    attrs = {
        "target": attr.label(
            allow_files = True,
            executable = True,
            mandatory = True,
            cfg = _ynnpack_platform_transition,
        ),
        "platform": attr.label(
            mandatory = True,
        ),
    },
    executable = True,
)
