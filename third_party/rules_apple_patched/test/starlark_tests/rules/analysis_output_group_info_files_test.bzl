# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Starlark test rule for OutputGroupInfo output group files."""

load(
    "//test/starlark_tests/rules:analysis_provider_test.bzl",
    "make_provider_test_rule",
)
load(
    "//test/starlark_tests/rules:assertions.bzl",
    "assertions",
)

visibility("//test/starlark_tests/...")

def _get_output_group_files(ctx, provider):
    """Returns list of files from a given output group from OutputGroupInfo."""
    output_group_name = ctx.attr.output_group_name
    if not hasattr(provider, output_group_name):
        fail(
            "OutputGroupInfo does not have output group: %s, available output groups are: %s" % (
                output_group_name,
                [x for x in dir(provider) if x not in ["to_json", "to_proto"]],
            ),
        )

    return getattr(provider, output_group_name).to_list()

def _analysis_output_group_info_files_test_assertion(ctx, env, output_group_files):
    return assertions.contains_files(
        env = env,
        expected_files = ctx.attr.expected_outputs,
        actual_files = output_group_files,
    )

def make_analysis_output_group_info_files_test(config_settings = {}):
    return make_provider_test_rule(
        provider = OutputGroupInfo,
        provider_fn = _get_output_group_files,
        assertion_fn = _analysis_output_group_info_files_test_assertion,
        attrs = {
            "output_group_name": attr.string(
                mandatory = True,
                doc = "Name of the output group to source files from.",
            ),
            "expected_outputs": attr.string_list(
                mandatory = True,
                doc = "List of relative output file paths expected as outputs of the output group.",
            ),
        },
        config_settings = {
            "//command_line_option:objc_generate_linkmap": "true",  # output_group: linkmaps
            "//command_line_option:apple_generate_dsym": "true",  # output_group: dsyms
            "//command_line_option:macos_cpus": "arm64,x86_64",
            "//command_line_option:ios_multi_cpus": "sim_arm64,x86_64",
            "//command_line_option:tvos_cpus": "sim_arm64,x86_64",
            "//command_line_option:visionos_cpus": "sim_arm64",
            "//command_line_option:watchos_cpus": "arm64,x86_64",
        } | config_settings,
    )

analysis_output_group_info_files_test = make_analysis_output_group_info_files_test()
