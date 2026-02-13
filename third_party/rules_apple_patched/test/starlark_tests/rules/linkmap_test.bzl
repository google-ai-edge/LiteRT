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

"""Starlark test rules for linkmap generation."""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)

def _linkmap_test_impl(ctx):
    """Implementation of the linkmap_test rule."""
    env = analysistest.begin(ctx)
    target_under_test = ctx.attr.target_under_test[0]
    architectures = ctx.attr.architectures

    if not architectures:
        architectures = ["arm64", "x86_64"]

    outputs = {
        x.short_path: None
        for x in target_under_test[DefaultInfo].files.to_list()
    }

    package = target_under_test.label.package
    expected_linkmap_names = ctx.attr.expected_linkmap_names
    if not expected_linkmap_names:
        expected_linkmap_names = [target_under_test.label.name]

    expected_linkmaps = []
    for expected_linkmap_name in expected_linkmap_names:
        for architecture in architectures:
            linkmap_filename = paths.join(
                package,
                "{}_{}.linkmap".format(expected_linkmap_name, architecture),
            )
            expected_linkmaps.append(linkmap_filename)

    for expected in expected_linkmaps:
        asserts.true(
            env,
            expected in outputs,
            msg = "Expected\n\n{0}\n\nto be built. Contents were:\n\n{1}\n\n".format(
                expected,
                "\n".join(outputs.keys()),
            ),
        )

    return analysistest.end(env)

linkmap_test = analysistest.make(
    _linkmap_test_impl,
    attrs = {
        "architectures": attr.string_list(
            mandatory = False,
            default = [],
            doc = """
List of architectures to verify for the given dSYM bundles as provided. Defaults to x86_64 for all
platforms.
""",
        ),
        "expected_linkmap_names": attr.string_list(
            mandatory = False,
            default = [],
            doc = """
List of linkmap names to verify that linkmaps are created. Defaults to the target name if none is
provided.
""",
        ),
    },
    config_settings = {
        "//command_line_option:objc_generate_linkmap": "true",
        "//command_line_option:macos_cpus": "arm64,x86_64",
        "//command_line_option:ios_multi_cpus": "sim_arm64,x86_64",
        "//command_line_option:tvos_cpus": "sim_arm64,x86_64",
        "//command_line_option:visionos_cpus": "sim_arm64",
        "//command_line_option:watchos_cpus": "arm64,x86_64",
    },
    fragments = ["apple"],
)
