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
"""A rule that transitions a target to build_include=cpu_only."""

_BUILD_INCLUDE_SETTING = "//litert/build_common:build_include"

def _cpu_only_transition_impl(_settings, _attr):
    return {
        _BUILD_INCLUDE_SETTING: "cpu_only",
    }

_cpu_only_transition = transition(
    implementation = _cpu_only_transition_impl,
    inputs = [],
    outputs = [
        _BUILD_INCLUDE_SETTING,
    ],
)

def _cpu_only_target_impl(ctx):
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

cpu_only_target = rule(
    implementation = _cpu_only_target_impl,
    doc = "Builds `target` with `build_include=cpu_only`.",
    attrs = {
        "target": attr.label(
            allow_files = True,
            executable = True,
            mandatory = True,
            cfg = _cpu_only_transition,
        ),
    },
    executable = True,
)
