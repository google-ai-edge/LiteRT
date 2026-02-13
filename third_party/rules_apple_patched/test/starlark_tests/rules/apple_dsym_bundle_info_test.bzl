# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""Starlark test rules for debug symbols."""

load(
    "//apple:providers.bzl",
    "AppleDsymBundleInfo",
)
load(
    "//test/starlark_tests/rules:analysis_provider_test.bzl",
    "make_provider_test_rule",
)
load(
    "//test/starlark_tests/rules:assertions.bzl",
    "assertions",
)

visibility("//test/starlark_tests/...")

def _assert_contains_expected_direct_and_transitive_dsyms(
        ctx,
        env,
        apple_dsym_bundle_info):
    """Assert AppleDsymBundleInfo contains expected direct and transitive dsyms."""
    assertions.contains_files(
        env = env,
        expected_files = ctx.attr.expected_direct_dsyms,
        actual_files = apple_dsym_bundle_info.direct_dsyms,
    )
    assertions.contains_files(
        env = env,
        expected_files = ctx.attr.expected_transitive_dsyms,
        actual_files = apple_dsym_bundle_info.transitive_dsyms.to_list(),
    )

apple_dsym_bundle_info_test = make_provider_test_rule(
    provider = AppleDsymBundleInfo,
    assertion_fn = _assert_contains_expected_direct_and_transitive_dsyms,
    attrs = {
        "expected_direct_dsyms": attr.string_list(
            mandatory = True,
            doc = """
List of bundle names in the format <bundle_name>.<bundle_extension> to verify that dSYM bundles are
created for them as direct dependencies of the given providers.
""",
        ),
        "expected_transitive_dsyms": attr.string_list(
            mandatory = True,
            doc = """
List of bundle names in the format <bundle_name>.<bundle_extension> to verify that dSYM bundles are
created for them as transitive dependencies of the given providers.
""",
        ),
    },
    config_settings = {
        "//command_line_option:apple_generate_dsym": "true",
        "//command_line_option:macos_cpus": "arm64,x86_64",
        "//command_line_option:ios_multi_cpus": "sim_arm64,x86_64",
        "//command_line_option:tvos_cpus": "sim_arm64,x86_64",
        "//command_line_option:visionos_cpus": "sim_arm64",
        "//command_line_option:watchos_cpus": "arm64,x86_64",
    },
)
