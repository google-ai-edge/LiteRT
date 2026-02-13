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

"""tvos_static_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)
load(
    ":common.bzl",
    "common",
)

def tvos_static_framework_test_suite(name):
    """Test suite for tvos_static_framework.

    Args:
      name: the base name to be used in things created by this macro
    """
    archive_contents_test(
        name = "{}_contents_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:static_fmwk",
        contains = [
            "$BUNDLE_ROOT/Headers/static_fmwk.h",
            "$BUNDLE_ROOT/Headers/shared.h",
            "$BUNDLE_ROOT/Modules/module.modulemap",
        ],
        tags = [name],
    )

    # Tests Swift tvos_static_framework builds correctly for sim_arm64, and x86_64 cpu's.
    archive_contents_test(
        name = "{}_swift_sim_arm64_builds_using_tvos_cpus".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:swift_static_fmwk",
        cpus = {
            "tvos_cpus": ["x86_64", "sim_arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/swift_static_fmwk",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_tvos.arm_sim_support, "platform TVOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_TVOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_x86_64_builds_using_tvos_cpus".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:swift_static_fmwk",
        cpus = {
            "tvos_cpus": ["x86_64", "sim_arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/swift_static_fmwk",
        binary_test_architecture = "x86_64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "minos " + common.min_os_tvos.baseline, "platform TVOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_TVOS"],
        tags = [name],
    )

    # Tests Swift tvos_static_framework builds correctly using apple_platforms.
    archive_contents_test(
        name = "{}_swift_sim_arm64_builds_using_platforms".format(name),
        apple_platforms = [
            "@build_bazel_apple_support//platforms:tvos_sim_arm64",
            "@build_bazel_apple_support//platforms:tvos_x86_64",
        ],
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:swift_static_fmwk",
        binary_test_file = "$BUNDLE_ROOT/swift_static_fmwk",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_TVOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_x86_64_builds_using_platforms".format(name),
        apple_platforms = [
            "@build_bazel_apple_support//platforms:tvos_sim_arm64",
            "@build_bazel_apple_support//platforms:tvos_x86_64",
        ],
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:swift_static_fmwk",
        binary_test_file = "$BUNDLE_ROOT/swift_static_fmwk",
        binary_test_architecture = "x86_64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform TVOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_TVOS"],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
