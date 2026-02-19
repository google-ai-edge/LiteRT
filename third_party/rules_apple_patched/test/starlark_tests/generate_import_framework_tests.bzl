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

"""generate_import_framework Starlark tests."""

load(
    "//test/starlark_tests/rules:analysis_target_outputs_test.bzl",
    "analysis_target_outputs_test",
    "make_analysis_target_outputs_test",
)

analysis_target_outputs_with_ios_platform_test = make_analysis_target_outputs_test(
    config_settings = {
        "//command_line_option:ios_multi_cpus": "x86_64",
    },
)

def generate_import_framework_test_suite(name):
    """Test suite for generate_import_framework.

    Args:
      name: the base name to be used in things created by this macro
    """
    analysis_target_outputs_test(
        name = "{}_dynamic_frameworks_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:iOSDynamicFramework",
        expected_outputs = [
            "iOSDynamicFramework-intermediates/iOSDynamicFramework.framework/iOSDynamicFramework",
            "iOSDynamicFramework-intermediates/iOSDynamicFramework.framework/Info.plist",
            "iOSDynamicFramework-intermediates/iOSDynamicFramework.framework/Headers/SharedClass.h",
            "iOSDynamicFramework-intermediates/iOSDynamicFramework.framework/Headers/iOSDynamicFramework.h",
            "iOSDynamicFramework-intermediates/iOSDynamicFramework.framework/Modules/module.modulemap",
            "iOSDynamicFramework-intermediates/iOSDynamicFramework.framework/Resources/iOSDynamicFramework.bundle/Info.plist",
        ],
        tags = [name],
    )

    analysis_target_outputs_test(
        name = "{}_static_frameworks_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:iOSStaticFramework",
        expected_outputs = [
            "iOSStaticFramework-intermediates/iOSStaticFramework.framework/Headers/SharedClass.h",
            "iOSStaticFramework-intermediates/iOSStaticFramework.framework/Headers/iOSStaticFramework.h",
            "iOSStaticFramework-intermediates/iOSStaticFramework.framework/Modules/module.modulemap",
            "iOSStaticFramework-intermediates/iOSStaticFramework.framework/Resources/iOSStaticFramework.bundle/Info.plist",
            "iOSStaticFramework-intermediates/iOSStaticFramework.framework/iOSStaticFramework",
            "iOSStaticFramework-intermediates/iOSStaticFramework.framework/Info.plist",
        ],
        tags = [name],
    )

    analysis_target_outputs_with_ios_platform_test(
        name = "{}_swift_static_frameworks_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:iOSSwiftStaticFramework",
        expected_outputs = [
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/iOSSwiftStaticFramework",
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/Info.plist",
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/Headers/swift_library-Swift.h",
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/Headers/iOSSwiftStaticFramework.h",
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/Modules/module.modulemap",
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/Modules/iOSSwiftStaticFramework.swiftmodule/x86_64.swiftinterface",
            "iOSSwiftStaticFramework-intermediates/iOSSwiftStaticFramework.framework/Modules/iOSSwiftStaticFramework.swiftmodule/x86_64-apple-ios-simulator.swiftinterface",
        ],
        tags = [name],
    )

    analysis_target_outputs_with_ios_platform_test(
        name = "{}_swift_without_module_interfaces_static_frameworks_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/ios:iOSSwiftStaticFrameworkWithoutModuleInterfaces",
        expected_outputs = [
            "iOSSwiftStaticFrameworkWithoutModuleInterfaces-intermediates/iOSSwiftStaticFrameworkWithoutModuleInterfaces.framework/iOSSwiftStaticFrameworkWithoutModuleInterfaces",
            "iOSSwiftStaticFrameworkWithoutModuleInterfaces-intermediates/iOSSwiftStaticFrameworkWithoutModuleInterfaces.framework/Info.plist",
            "iOSSwiftStaticFrameworkWithoutModuleInterfaces-intermediates/iOSSwiftStaticFrameworkWithoutModuleInterfaces.framework/Headers/swift_library-Swift.h",
            "iOSSwiftStaticFrameworkWithoutModuleInterfaces-intermediates/iOSSwiftStaticFrameworkWithoutModuleInterfaces.framework/Headers/iOSSwiftStaticFrameworkWithoutModuleInterfaces.h",
            "iOSSwiftStaticFrameworkWithoutModuleInterfaces-intermediates/iOSSwiftStaticFrameworkWithoutModuleInterfaces.framework/Modules/module.modulemap",
        ],
        tags = [name],
    )

    analysis_target_outputs_test(
        name = "{}_versioned_frameworks_outputs_test".format(name),
        target_under_test = "//test/starlark_tests/targets_under_test/macos:generated_macos_dynamic_versioned_fmwk",
        expected_outputs = [
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Headers/SharedClass.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Headers/generated_macos_dynamic_versioned_fmwk.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Modules/module.modulemap",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Resources/Info.plist",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Headers/SharedClass.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Headers/generated_macos_dynamic_versioned_fmwk.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Modules/module.modulemap",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/Resources/Info.plist",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/A/generated_macos_dynamic_versioned_fmwk",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Headers/SharedClass.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Headers/generated_macos_dynamic_versioned_fmwk.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Modules/module.modulemap",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/Resources/Info.plist",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/B/generated_macos_dynamic_versioned_fmwk",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current/Headers/SharedClass.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current/Headers/generated_macos_dynamic_versioned_fmwk.h",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current/Modules/module.modulemap",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current/Resources/Info.plist",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/Versions/Current/generated_macos_dynamic_versioned_fmwk",
            "generated_macos_dynamic_versioned_fmwk-intermediates/generated_macos_dynamic_versioned_fmwk.framework/generated_macos_dynamic_versioned_fmwk",
        ],
        tags = [name],
    )

    native.test_suite(
        name = name,
        tags = [name],
    )
