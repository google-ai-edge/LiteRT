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

"""
Starlark helpers to build LiteRT targets under a specific build flavor
without requiring users to pass --//litert/build_common:build_include=...

Provides a flavored_cc_alias rule that forwards providers from the
"deps" label list but builds those dependencies with a configuration
transition that sets //litert/build_common:build_include accordingly.
"""

def _flavor_transition_impl(settings, attr):
    _ignore = settings  # buildifier: disable=unused-variable
    return {"//litert/build_common:build_include": attr.build_include}

flavor_transition = transition(
    implementation = _flavor_transition_impl,
    inputs = [],
    outputs = ["//litert/build_common:build_include"],
)

def _flavored_cc_alias_impl(ctx):
    dep = ctx.attr.deps[0]
    d = dep[DefaultInfo]
    default = DefaultInfo(
        files = d.files,
        data_runfiles = d.data_runfiles,
        default_runfiles = d.default_runfiles,
    )
    result = [default]
    if CcInfo in dep:
        result.append(dep[CcInfo])
    if OutputGroupInfo in dep:
        result.append(dep[OutputGroupInfo])
    return result

flavored_cc_alias = rule(
    implementation = _flavored_cc_alias_impl,
    attrs = {
        "deps": attr.label_list(cfg = flavor_transition, mandatory = True),
        "build_include": attr.string(mandatory = True),
    },
    fragments = ["cpp"],
)
