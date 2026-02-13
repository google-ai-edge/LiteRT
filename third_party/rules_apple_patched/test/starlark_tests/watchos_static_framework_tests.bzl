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

"""watchos_static_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

def watchos_static_framework_test_suite(name):
    """Test suite for watchos_static_framework.

    Args:
      name: the base name to be used in things created by this macro
    """

    archive_contents_test(
        name = "{}_contents_test".format(name),
        build_type = "simulator",
        compilation_mode = "opt",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:static_fmwk",
        contains = [
            "$BUNDLE_ROOT/Headers/static_fmwk.h",
            "$BUNDLE_ROOT/Headers/shared.h",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/static_fmwk",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_angle_bracketed_import_in_umbrella_header".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:static_fmwk",
        text_test_file = "$BUNDLE_ROOT/Headers/static_fmwk.h",
        text_test_values = ["#import <static_fmwk/shared.h>"],
        tags = [name],
    )

    # Tests Swift watchos_static_framework builds correctly for arm64, and i386 cpu's.
    archive_contents_test(
        name = "{}_swift_arm64_builds".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:swift_static_fmwk",
        cpus = {
            "watchos_cpus": ["x86_64", "arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/swift_static_fmwk",
        binary_test_architecture = "arm64",
        macho_load_commands_contain = ["cmd LC_BUILD_VERSION", "platform WATCHOSSIMULATOR"],
        macho_load_commands_not_contain = ["cmd LC_VERSION_MIN_WATCHOS"],
        tags = [name],
    )
    archive_contents_test(
        name = "{}_swift_x86_64_builds".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:swift_static_fmwk",
        cpus = {
            "watchos_cpus": ["x86_64", "arm64"],
        },
        binary_test_file = "$BUNDLE_ROOT/swift_static_fmwk",
        binary_test_architecture = "x86_64",
        macho_load_commands_contain = ["cmd LC_VERSION_MIN_WATCHOS"],
        macho_load_commands_not_contain = ["cmd LC_BUILD_VERSION", "platform WATCHOSSIMULATOR"],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
