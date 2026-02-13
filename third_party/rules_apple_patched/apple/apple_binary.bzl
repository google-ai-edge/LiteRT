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

"""Starlark implementation of `apple_binary` to transition from native Bazel."""

load(
    "//apple/internal:linking_support.bzl",
    "linking_support",
)
load(
    "//apple/internal:providers.bzl",
    "AppleExecutableBinaryInfo",
    "new_appleexecutablebinaryinfo",
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

def _linker_flag_for_sdk_dylib(dylib):
    """Returns a linker flag suitable for linking the given `sdk_dylib` value.

    As does Bazel core, we strip a leading `lib` if it is present in the name
    of the library.

    Args:
        dylib: The name of the library, as specified in the `sdk_dylib`
            attribute.

    Returns:
        A linker flag used to link to the given library.
    """
    if dylib.startswith("lib"):
        dylib = dylib[3:]
    return "-l{}".format(dylib)

def _apple_binary_impl(ctx):
    if ctx.attr.platform_type == "visionos":
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
        if xcode_version_config.xcode_version() < apple_common.dotted_version("15.1"):
            fail("""
visionOS binaries require a visionOS SDK provided by Xcode 15.1 beta or later.

Resolved Xcode is version {xcode_version}.
""".format(xcode_version = str(xcode_version_config.xcode_version())))

    binary_type = ctx.attr.binary_type
    bundle_loader = ctx.attr.bundle_loader
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder

    extra_linkopts = []

    if binary_type == "dylib":
        extra_linkopts.append("-dynamiclib")
    elif binary_type == "loadable_bundle":
        extra_linkopts.extend([
            "-bundle",
            "-Wl,-rpath,@loader_path/Frameworks",
        ])

    extra_linkopts.extend([
        _linker_flag_for_sdk_dylib(dylib)
        for dylib in ctx.attr.sdk_dylibs
    ])
    extra_linkopts.extend([
        "-Wl,-framework,{}".format(framework)
        for framework in ctx.attr.sdk_frameworks
    ])
    extra_linkopts.extend([
        "-Wl,-weak_framework,{}".format(framework)
        for framework in ctx.attr.weak_sdk_frameworks
    ])

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        bundle_loader = bundle_loader,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = extra_linkopts,
        platform_prerequisites = None,
        rule_descriptor = None,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary

    providers = [
        DefaultInfo(
            files = depset([binary_artifact]),
            runfiles = ctx.runfiles(
                collect_data = True,
                collect_default = True,
                files = [binary_artifact] + ctx.files.data,
            ),
        ),
        OutputGroupInfo(**link_result.output_groups),
        coverage_common.instrumented_files_info(
            ctx,
            dependency_attributes = ["bundle_loader", "deps"],
        ),
        link_result.debug_outputs_provider,
    ]

    # If the binary was an executable, also propagate the appropriate provider
    # so that bundles can use it as their loader.
    if binary_type == "executable":
        providers.append(
            new_appleexecutablebinaryinfo(
                binary = binary_artifact,
                cc_info = link_result.cc_info,
            ),
        )

    return providers

apple_binary = rule_factory.create_apple_rule(
    doc = """
This rule produces single- or multi-architecture ("fat") binaries targeting
Apple platforms.

The `lipo` tool is used to combine files of multiple architectures. One of
several flags may control which architectures are included in the output,
depending on the value of the `platform_type` attribute.

NOTE: In most situations, users should prefer the platform- and
product-type-specific rules, such as `macos_command_line_application`. This
rule is being provided for the purpose of transitioning users from the built-in
implementation of `apple_binary` in Bazel core so that it can be removed.
""",
    implementation = _apple_binary_impl,
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.platform_attrs(),
        {
            "binary_type": attr.string(
                default = "executable",
                doc = """
The type of binary that this target should build. Option are:

*   `dylib`: The output binary is meant to be loaded at load time (when
    the operating system is loading the binary into memory) and cannot
    be unloaded.
*   `executable` (default): The output binary is an executable and must
    implement the `main` function.
*   `loadable_bundle`: The output binary is a loadable bundle that may be
    loaded at runtime. When building a bundle, you may also pass a
    `bundle_loader` binary that is an executable that contains symbols
    referenced by but not implemented in the loadable bundle.
""",
                values = ["dylib", "executable", "loadable_bundle"],
            ),
            "bundle_loader": attr.label(
                doc = """
The target representing the executable that will be loading this bundle.
Undefined symbols from the bundle are checked against this executable during
linking as if it were one of the dynamic libraries the bundle was linked with.
""",
                providers = [AppleExecutableBinaryInfo],
            ),
            "data": attr.label_list(allow_files = True),
            "sdk_dylibs": attr.string_list(
                allow_empty = True,
                doc = """
Names of SDK `.dylib` libraries to link with (e.g., `libz` or `libarchive`).
`libc++` is included automatically if the binary has any C++ or Objective-C++
sources in its dependency tree. When linking a binary, all libraries named in
that binary's transitive dependency graph are used.
""",
            ),
            "sdk_frameworks": attr.string_list(
                allow_empty = True,
                doc = """
Names of SDK frameworks to link with (e.g., `AddressBook`, `QuartzCore`).
`UIKit` and `Foundation` are always included, even if this attribute is
provided and does not list them.

This attribute is discouraged; in general, targets should list system
framework dependencies in the library targets where that framework is used,
not in the top-level bundle.
""",
            ),
            "weak_sdk_frameworks": attr.string_list(
                allow_empty = True,
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
