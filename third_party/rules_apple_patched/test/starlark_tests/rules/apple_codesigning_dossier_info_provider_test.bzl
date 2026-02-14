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

"""Starlark test rules for debug symbols."""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)
load(
    "//apple:providers.bzl",
    "AppleCodesigningDossierInfo",
)
load(
    "//test/starlark_tests/rules:analysis_provider_test.bzl",
    "make_provider_test_rule",
)

visibility("//test/starlark_tests/...")

def _assert_contains_expected_dossier(
        ctx,
        env,
        apple_codesigning_dossier_info):
    """Assert AppleCodesigningDossierInfo contains the expected dossier zip."""

    target_under_test = analysistest.target_under_test(env)

    dossier_package_relative_path = paths.relativize(
        apple_codesigning_dossier_info.dossier.short_path,
        target_under_test.label.package,
    )

    asserts.equals(
        env,
        dossier_package_relative_path,
        ctx.attr.expected_dossier,
        "{} not contained in {}".format(
            dossier_package_relative_path,
            ctx.attr.expected_dossier,
        ),
    )

apple_codesigning_dossier_info_provider_test = make_provider_test_rule(
    provider = AppleCodesigningDossierInfo,
    assertion_fn = _assert_contains_expected_dossier,
    attrs = {
        "expected_dossier": attr.string(
            mandatory = True,
            doc = "String to indicate the dossier we expect to have generated for the target.",
        ),
    },
    config_settings = {
        "//command_line_option:macos_cpus": "arm64,x86_64",
        "//command_line_option:ios_multi_cpus": "sim_arm64,x86_64",
        "//command_line_option:tvos_cpus": "sim_arm64,x86_64",
        "//command_line_option:visionos_cpus": "sim_arm64",
        "//command_line_option:watchos_cpus": "arm64,x86_64",
    },
)
