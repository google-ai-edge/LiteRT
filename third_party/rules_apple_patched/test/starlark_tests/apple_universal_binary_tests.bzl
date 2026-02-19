# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""apple_universal_binary Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_target_outputs_test.bzl",
    "analysis_target_outputs_test",
)
load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "binary_contents_test",
)

def apple_universal_binary_test_suite(name):
    """Test suite for apple_universal_binary.

    Args:
      name: the base name to be used in things created by this macro
    """
    analysis_target_outputs_test(
        name = "{}_output_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/apple:multi_arch_cc_binary",
        expected_outputs = ["multi_arch_cc_binary"],
        tags = [name],
    )

    binary_contents_test(
        name = "{}_x86_binary_contents_test".format(name),
        build_type = "device",
        cpus = {"macos_cpus": ["x86_64", "arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/apple:multi_arch_cc_binary",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["__Z19function_for_x86_64v"],
        binary_not_contains_symbols = ["__Z19function_for_arch64v"],
        tags = [name],
    )

    binary_contents_test(
        name = "{}_arm64_binary_contents_test".format(name),
        build_type = "device",
        cpus = {"macos_cpus": ["x86_64", "arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/apple:multi_arch_cc_binary",
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        binary_contains_symbols = ["__Z19function_for_arch64v"],
        binary_not_contains_symbols = ["__Z19function_for_x86_64v"],
        tags = [name],
    )

    # The following two tests verify the `forced_cpus` behavior. They set the
    # `macos_cpus` to be a single CPU (the opposite CPU from the one being
    # checked in each test), then verify that the other slice still exists (thus
    # `forced_cpus` overrode the `--macos_cpus` flag correctly).
    binary_contents_test(
        name = "{}_forced_cpus_x86_binary_contents_test".format(name),
        build_type = "device",
        cpus = {"macos_cpus": ["arm64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/apple:multi_arch_forced_cpus_cc_binary",
        binary_test_file = "$BINARY",
        binary_test_architecture = "x86_64",
        binary_contains_symbols = ["__Z19function_for_x86_64v"],
        binary_not_contains_symbols = ["__Z19function_for_arch64v"],
        tags = [name],
    )

    binary_contents_test(
        name = "{}_forced_cpus_arm64_binary_contents_test".format(name),
        build_type = "device",
        cpus = {"macos_cpus": ["x86_64"]},
        target_under_test = "//test/starlark_tests/targets_under_test/apple:multi_arch_forced_cpus_cc_binary",
        binary_test_file = "$BINARY",
        binary_test_architecture = "arm64",
        binary_contains_symbols = ["__Z19function_for_arch64v"],
        binary_not_contains_symbols = ["__Z19function_for_x86_64v"],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
