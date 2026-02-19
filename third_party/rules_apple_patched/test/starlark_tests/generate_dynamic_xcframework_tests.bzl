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

"""generate_dynamic_xcframework Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_target_outputs_test.bzl",
    "analysis_target_outputs_test",
)

def generate_dynamic_xcframework_test_suite(name):
    """Test suite for generate_dynamic_xcframework.

    Args:
      name: the base name to be used in things created by this macro
    """
    analysis_target_outputs_test(
        name = "{}_ios_frameworks_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:generated_dynamic_xcframework_with_headers",
        expected_outputs = [
            "generated_dynamic_xcframework_with_headers.xcframework/Info.plist",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_x86_64-simulator/generated_dynamic_xcframework_with_headers.framework/generated_dynamic_xcframework_with_headers",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_x86_64-simulator/generated_dynamic_xcframework_with_headers.framework/Info.plist",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_x86_64-simulator/generated_dynamic_xcframework_with_headers.framework/Headers/SharedClass.h",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_x86_64-simulator/generated_dynamic_xcframework_with_headers.framework/Headers/generated_dynamic_xcframework_with_headers.h",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_x86_64-simulator/generated_dynamic_xcframework_with_headers.framework/Modules/module.modulemap",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_arm64e/generated_dynamic_xcframework_with_headers.framework/generated_dynamic_xcframework_with_headers",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_arm64e/generated_dynamic_xcframework_with_headers.framework/Info.plist",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_arm64e/generated_dynamic_xcframework_with_headers.framework/Headers/SharedClass.h",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_arm64e/generated_dynamic_xcframework_with_headers.framework/Headers/generated_dynamic_xcframework_with_headers.h",
            "generated_dynamic_xcframework_with_headers.xcframework/ios-arm64_arm64e/generated_dynamic_xcframework_with_headers.framework/Modules/module.modulemap",
        ],
        tags = [name],
    )

    analysis_target_outputs_test(
        name = "{}_macos_versioned_frameworks_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:generated_dynamic_macos_versioned_xcframework",
        expected_outputs = [
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Headers/SharedClass.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Headers/generated_dynamic_macos_versioned_xcframework.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Modules/module.modulemap",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Resources/Info.plist",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Headers/SharedClass.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Headers/generated_dynamic_macos_versioned_xcframework.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Modules/module.modulemap",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/Resources/Info.plist",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/A/generated_dynamic_macos_versioned_xcframework",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Headers/SharedClass.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Headers/generated_dynamic_macos_versioned_xcframework.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Modules/module.modulemap",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/Resources/Info.plist",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/B/generated_dynamic_macos_versioned_xcframework",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current/Headers/SharedClass.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current/Headers/generated_dynamic_macos_versioned_xcframework.h",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current/Modules/module.modulemap",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current/Resources/Info.plist",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/Versions/Current/generated_dynamic_macos_versioned_xcframework",
            "generated_dynamic_macos_versioned_xcframework.xcframework/macos-arm64_arm64e_x86_64/generated_dynamic_macos_versioned_xcframework.framework/generated_dynamic_macos_versioned_xcframework",
        ],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
