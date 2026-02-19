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

"""watchos_dynamic_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_failure_message_test.bzl",
    "analysis_failure_message_test",
)
load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)
load(
    "//test/starlark_tests/rules:infoplist_contents_test.bzl",
    "infoplist_contents_test",
)

def watchos_dynamic_framework_test_suite(name):
    """Test suite for watchos_dynamic_framework.

    Args:
      name: the base name to be used in things created by this macro
    """

    archive_contents_test(
        name = "{}_archive_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:basic_framework",
        contains = [
            "$BUNDLE_ROOT/BasicFramework",
            "$BUNDLE_ROOT/Headers/BasicFramework.h",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/Modules/BasicFramework.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/Modules/BasicFramework.swiftmodule/x86_64.swiftmodule",
        ],
        tags = [name],
    )

    infoplist_contents_test(
        name = "{}_plist_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:basic_framework",
        expected_values = {
            "BuildMachineOSBuild": "*",
            "CFBundleExecutable": "BasicFramework",
            "CFBundleIdentifier": "com.google.example.framework",
            "CFBundleName": "BasicFramework",
            "CFBundleSupportedPlatforms:0": "WatchSimulator*",
            "DTCompiler": "com.apple.compilers.llvm.clang.1_0",
            "DTPlatformBuild": "*",
            "DTPlatformName": "watchsimulator*",
            "DTPlatformVersion": "*",
            "DTSDKBuild": "*",
            "DTSDKName": "watchsimulator*",
            "DTXcode": "*",
            "DTXcodeBuild": "*",
            "MinimumOSVersion": "6.0",
            "UIDeviceFamily:0": "4",
        },
        tags = [name],
    )

    archive_contents_test(
        name = "{}_direct_dependency_archive_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:basic_framework_with_direct_dependency",
        contains = [
            "$BUNDLE_ROOT/DirectDependencyTest",
            "$BUNDLE_ROOT/Headers/DirectDependencyTest.h",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/Modules/DirectDependencyTest.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/Modules/DirectDependencyTest.swiftmodule/x86_64.swiftmodule",
        ],
        tags = [name],
    )

    archive_contents_test(
        name = "{}_transitive_dependency_archive_contents_test".format(name),
        build_type = "simulator",
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:basic_framework_with_transitive_dependency",
        contains = [
            "$BUNDLE_ROOT/TransitiveDependencyTest",
            "$BUNDLE_ROOT/Headers/TransitiveDependencyTest.h",
            "$BUNDLE_ROOT/Info.plist",
            "$BUNDLE_ROOT/Modules/module.modulemap",
            "$BUNDLE_ROOT/Modules/TransitiveDependencyTest.swiftmodule/x86_64.swiftdoc",
            "$BUNDLE_ROOT/Modules/TransitiveDependencyTest.swiftmodule/x86_64.swiftmodule",
        ],
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_multiple_deps_in_dynamic_framework_fail".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:dynamic_fmwk_with_multiple_dependencies",
        expected_error = """\
    error: Swift dynamic frameworks expect a single swift_library dependency.
    """,
        tags = [name],
    )

    analysis_failure_message_test(
        name = "{}_non_swiftlib_dep_in_dynamic_framework_fail".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/watchos:dynamic_fmwk_with_objc_library",
        expected_error = """\
    error: Swift dynamic frameworks expect a single swift_library dependency.
    """,
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
