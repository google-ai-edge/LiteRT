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

"""Swift-specific `tvos_application` bundling tests."""

load(
    "//test/starlark_tests/rules:common_verification_tests.bzl",
    "archive_contents_test",
)

def tvos_application_swift_test_suite(name):
    """Test suite for tvos_application_swift.

    Args:
      name: the base name to be used in things created by this macro
    """

    # If an app is built with a min OS before ABI stability, targeting the
    # simulator, the Swift runtime should be bundled in the Frameworks
    # directory but not in the IPA's root SwiftSupport directory.
    archive_contents_test(
        name = "{}_simulator_build_has_swift_libs_in_frameworks_dir_only_test".format(name),
        build_type = "simulator",
        contains = [
            "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
        ],
        not_contains = [
            "$ARCHIVE_ROOT/SwiftSupport/appletvsimulator/libswiftCore.dylib",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_direct_swift_dep",
        tags = [name],
    )

    # If an app is built with a min OS before ABI stability, targeting the
    # device, the Swift runtime should be bundled in both the Frameworks
    # directory and the IPA's root SwiftSupport directory.
    archive_contents_test(
        name = "{}_device_build_has_swift_libs_in_frameworks_and_support_dirs_test".format(name),
        build_type = "device",
        contains = [
            "$ARCHIVE_ROOT/SwiftSupport/appletvos/libswiftCore.dylib",
            "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_direct_swift_dep",
        tags = [name],
    )

    # If an app is built with a min OS after ABI stability, the Swift runtime
    # should not be bundled at all.
    archive_contents_test(
        name = "{}_build_for_stable_abi_does_not_have_swift_libs_test".format(name),
        build_type = "device",
        not_contains = [
            "$ARCHIVE_ROOT/SwiftSupport/appletvsimulator/libswiftCore.dylib",
            "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_direct_swift_dep_stable_abi",
        tags = [name],
    )

    # Make sure Swift runtime bundling works even if the dependency is
    # indirect (e.g., through an `objc_library`).
    archive_contents_test(
        name = "{}_swift_libs_are_bundled_through_indirect_deps_test".format(name),
        build_type = "device",
        contains = [
            "$ARCHIVE_ROOT/SwiftSupport/appletvos/libswiftCore.dylib",
            "$BUNDLE_ROOT/Frameworks/libswiftCore.dylib",
        ],
        target_under_test = "//test/starlark_tests/targets_under_test/tvos:app_with_indirect_swift_dep",
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
