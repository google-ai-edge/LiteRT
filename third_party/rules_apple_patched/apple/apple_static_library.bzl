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

"""apple_static_library Starlark implementation"""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple:providers.bzl",
    "ApplePlatformInfo",
)
load(
    "//apple/internal:linking_support.bzl",
    "linking_support",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebinaryinfo",
)
load(
    "//apple/internal:rule_attrs.bzl",
    "rule_attrs",
)
load(
    "//apple/internal:rule_factory.bzl",
    "rule_factory",
)
load(
    "//apple/internal:transition_support.bzl",
    "transition_support",
)

def _apple_static_library_impl(ctx):
    if ctx.attr.platform_type == "visionos":
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
        if xcode_version_config.xcode_version() < apple_common.dotted_version("15.1"):
            fail("""
visionOS static libraries require a visionOS SDK provided by Xcode 15.1 beta or later.

Resolved Xcode is version {xcode_version}.
""".format(xcode_version = str(xcode_version_config.xcode_version())))

    # Most validation of the platform type and minimum version OS currently happens in
    # `transition_support.apple_platform_split_transition`, either implicitly through native
    # `dotted_version` or explicitly through `fail` on an unrecognized platform type value.

    # Validate that the resolved platform matches the platform_type attr.
    for toolchain_key, resolved_toolchain in ctx.split_attr._cc_toolchain_forwarder.items():
        if resolved_toolchain[ApplePlatformInfo].target_os != ctx.attr.platform_type:
            fail("""
ERROR: Unexpected resolved platform:
Expected Apple platform type of "{platform_type}", but that was not found in {toolchain_key}.
""".format(
                platform_type = ctx.attr.platform_type,
                toolchain_key = toolchain_key,
            ))

    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    archive_result = linking_support.register_static_library_archive_action(
        ctx = ctx,
        cc_toolchains = cc_toolchain_forwarder,
    )

    files_to_build = [archive_result.library]
    runfiles = ctx.runfiles(
        files = files_to_build,
        collect_default = True,
        collect_data = True,
    )

    providers = [
        DefaultInfo(files = depset(files_to_build), runfiles = runfiles),
        new_applebinaryinfo(
            binary = archive_result.library,
            infoplist = None,
        ),
        archive_result.output_groups,
    ]

    if archive_result.objc:
        providers.append(archive_result.objc)

    return providers

apple_static_library = rule_factory.create_apple_rule(
    cfg = None,
    doc = """
This rule produces single- or multi-architecture ("fat") static libraries targeting
Apple platforms.

The `lipo` tool is used to combine files of multiple architectures. One of
several flags may control which architectures are included in the output,
depending on the value of the `platform_type` attribute.

NOTE: In most situations, users should prefer the platform- and
product-type-specific rules, such as `apple_static_xcframework`. This
rule is being provided for the purpose of transitioning users from the built-in
implementation of `apple_static_library` in Bazel core so that it can be removed.
""",
    implementation = _apple_static_library_impl,
    predeclared_outputs = {
        "lipo_archive": "%{name}_lipo.a",
    },
    attrs = [
        rule_attrs.common_tool_attrs(),
        rule_attrs.static_library_archive_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        {
            "additional_linker_inputs": attr.label_list(
                # Flag required for compile_one_dependency
                flags = ["DIRECT_COMPILE_TIME_INPUT"],
                allow_files = True,
                doc = """
A list of input files to be passed to the linker.
""",
            ),
            "avoid_deps": attr.label_list(
                cfg = transition_support.apple_platform_split_transition,
                providers = [CcInfo],
                doc = """
A list of library targets on which this framework depends in order to compile, but the transitive
closure of which will not be linked into the framework's binary.
""",
            ),
            "data": attr.label_list(
                allow_files = True,
                doc = """
Files to be made available to the library archive upon execution.
""",
            ),
            "deps": attr.label_list(
                cfg = transition_support.apple_platform_split_transition,
                providers = [CcInfo],
                doc = """
A list of dependencies targets that will be linked into this target's binary. Any resources, such as
asset catalogs, that are referenced by those targets will also be transitively included in the final
bundle.
""",
            ),
            "linkopts": attr.string_list(
                doc = """
A list of strings representing extra flags that should be passed to the linker.
""",
            ),
            "minimum_os_version": attr.string(
                mandatory = True,
                doc = """
A required string indicating the minimum OS version supported by the target, represented as a
dotted version number (for example, "9.0").
""",
            ),
            "platform_type": attr.string(
                mandatory = True,
                doc = """
The target Apple platform for which to create a binary. This dictates which SDK
is used for compilation/linking and which flag is used to determine the
architectures to target. For example, if `ios` is specified, then the output
binaries/libraries will be created combining all architectures specified by
`--ios_multi_cpus`. Options are:

*   `ios`: architectures gathered from `--ios_multi_cpus`.
*   `macos`: architectures gathered from `--macos_cpus`.
*   `tvos`: architectures gathered from `--tvos_cpus`.
*   `visionos`: architectures gathered from `--visionos_cpus`.
*   `watchos`: architectures gathered from `--watchos_cpus`.
""",
            ),
            "sdk_frameworks": attr.string_list(
                doc = """
Names of SDK frameworks to link with (e.g., `AddressBook`, `QuartzCore`).
`UIKit` and `Foundation` are always included, even if this attribute is
provided and does not list them.

This attribute is discouraged; in general, targets should list system
framework dependencies in the library targets where that framework is used,
not in the top-level bundle.
""",
            ),
            "sdk_dylibs": attr.string_list(
                doc = """
Names of SDK `.dylib` libraries to link with (e.g., `libz` or `libarchive`).
`libc++` is included automatically if the binary has any C++ or Objective-C++
sources in its dependency tree. When linking a binary, all libraries named in
that binary's transitive dependency graph are used.
""",
            ),
            "weak_sdk_frameworks": attr.string_list(
                doc = """
Names of SDK frameworks to weakly link with (e.g., `MediaAccessibility`).
Unlike regularly linked SDK frameworks, symbols from weakly linked
frameworks do not cause the binary to fail to load if they are not present in
the version of the framework available at runtime.

This attribute is discouraged; in general, targets should list system
framework dependencies in the library targets where that framework is used,
not in the top-level bundle.
""",
            ),
        },
    ],
)
