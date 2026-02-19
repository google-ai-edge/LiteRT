# Copyright 2018 The Bazel Authors. All rights reserved.
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
# limitations under the Lice

"""Internal helper definitions used by macOS command line rules."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "//apple:providers.bzl",
    "AppleBinaryInfoplistInfo",
    "AppleBundleVersionInfo",
)
load(
    "//apple/internal:apple_product_type.bzl",
    "apple_product_type",
)
load(
    "//apple/internal:apple_toolchains.bzl",
    "AppleMacToolsToolchainInfo",
    "AppleXPlatToolsToolchainInfo",
)
load(
    "//apple/internal:bundling_support.bzl",
    "bundling_support",
)
load(
    "//apple/internal:features_support.bzl",
    "features_support",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:linking_support.bzl",
    "linking_support",
)
load(
    "//apple/internal:platform_support.bzl",
    "platform_support",
)
load(
    "//apple/internal:resource_actions.bzl",
    "resource_actions",
)
load(
    "//apple/internal:rule_attrs.bzl",
    "rule_attrs",
)
load(
    "//apple/internal:rule_support.bzl",
    "rule_support",
)

def _macos_binary_infoplist_impl(ctx):
    """Implementation of the internal `macos_command_line_infoplist` rule.

    This rule is an internal implementation detail of
    `macos_command_line_application` and should not be used directly by clients.
    It merges Info.plists as would occur for a bundle but then propagates an
    `objc` provider with the necessary linkopts to embed the plist in a binary.

    Args:
      ctx: The rule context.

    Returns:
      A `struct` containing the `objc` provider that should be propagated to a
      binary that should have this plist embedded.
    """
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.tool,
    )

    actions = ctx.actions
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = ""
    if ctx.attr.bundle_id or ctx.attr.base_bundle_id:
        bundle_id = bundling_support.bundle_full_id(
            base_bundle_id = ctx.attr.base_bundle_id,
            bundle_id = ctx.attr.bundle_id,
            bundle_id_suffix = ctx.attr.bundle_id_suffix,
            bundle_name = bundle_name,
            suffix_default = ctx.attr._bundle_id_suffix_default,
        )

    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    rule_label = ctx.label

    platform_prerequisites = platform_support.platform_prerequisites(
        apple_fragment = ctx.fragments.apple,
        build_settings = apple_xplat_toolchain_info.build_settings,
        config_vars = ctx.var,
        cpp_fragment = ctx.fragments.cpp,
        device_families = rule_descriptor.allowed_device_families,
        explicit_minimum_deployment_os = ctx.attr.minimum_deployment_os_version,
        explicit_minimum_os = ctx.attr.minimum_os_version,
        features = features,
        objc_fragment = ctx.fragments.objc,
        platform_type_string = ctx.attr.platform_type,
        uses_swift = False,  # No binary deps to check.
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )

    infoplists = ctx.files.infoplists
    if ctx.attr.version and AppleBundleVersionInfo in ctx.attr.version:
        version_found = True
    else:
        version_found = False

    if not bundle_id and not infoplists and not version_found:
        fail("Internal error: at least one of bundle_id, infoplists, or version " +
             "should have been provided")

    merged_infoplist = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = None,
        file_name = "Info.plist",
    )

    resource_actions.merge_root_infoplists(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_id = bundle_id,
        bundle_name = bundle_name,
        environment_plist = ctx.file._environment_plist,
        include_executable_name = False,
        input_plists = infoplists,
        launch_storyboard = None,
        output_discriminator = None,
        output_pkginfo = None,
        output_plist = merged_infoplist,
        platform_prerequisites = platform_prerequisites,
        plisttool = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo].plisttool,
        rule_descriptor = rule_descriptor,
        rule_label = rule_label,
        version = ctx.attr.version,
    )

    return linking_support.sectcreate_objc_provider(
        rule_label,
        "__TEXT",
        "__info_plist",
        merged_infoplist,
    ) + [AppleBinaryInfoplistInfo(infoplist = merged_infoplist)]

macos_binary_infoplist = rule(
    implementation = _macos_binary_infoplist_impl,
    attrs = dicts.add(
        rule_attrs.common_tool_attrs(),
        rule_attrs.signing_attrs(
            supports_capabilities = False,
            profile_extension = ".provisionprofile",  # Unused, but staying consistent with macOS.
        ),
        {
            "infoplists": attr.label_list(
                allow_files = [".plist"],
                mandatory = False,
                allow_empty = True,
            ),
            "minimum_deployment_os_version": attr.string(mandatory = False),
            "minimum_os_version": attr.string(mandatory = False),
            "platform_type": attr.string(
                default = str(apple_common.platform_type.macos),
            ),
            "_environment_plist": attr.label(
                allow_single_file = True,
                default = "//apple/internal:environment_plist_macos",
            ),
            "version": attr.label(providers = [[AppleBundleVersionInfo]]),
        },
    ),
    fragments = ["apple", "cpp", "objc"],
)

def _macos_command_line_launchdplist_impl(ctx):
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.tool,
    )

    actions = ctx.actions
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    rule_label = ctx.label
    launchdplists = ctx.files.launchdplists

    platform_prerequisites = platform_support.platform_prerequisites(
        apple_fragment = ctx.fragments.apple,
        build_settings = apple_xplat_toolchain_info.build_settings,
        config_vars = ctx.var,
        cpp_fragment = ctx.fragments.cpp,
        device_families = rule_descriptor.allowed_device_families,
        explicit_minimum_deployment_os = ctx.attr.minimum_deployment_os_version,
        explicit_minimum_os = ctx.attr.minimum_os_version,
        features = features,
        objc_fragment = ctx.fragments.objc,
        platform_type_string = ctx.attr.platform_type,
        uses_swift = False,  # No binary deps to check.
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )

    if not launchdplists:
        fail("Internal error: launchdplists should have been provided")

    merged_launchdplist = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = None,
        file_name = "Launchd.plist",
    )

    resource_actions.merge_resource_infoplists(
        actions = actions,
        bundle_id = None,
        bundle_name_with_extension = bundle_name + bundle_extension,
        input_files = launchdplists,
        output_discriminator = None,
        output_plist = merged_launchdplist,
        platform_prerequisites = platform_prerequisites,
        plisttool = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo].plisttool,
        rule_label = rule_label,
    )

    return linking_support.sectcreate_objc_provider(
        rule_label,
        "__TEXT",
        "__launchd_plist",
        merged_launchdplist,
    )

macos_command_line_launchdplist = rule(
    implementation = _macos_command_line_launchdplist_impl,
    attrs = dicts.add(
        rule_attrs.common_tool_attrs(),
        {
            "launchdplists": attr.label_list(
                allow_files = [".plist"],
                mandatory = False,
            ),
            "minimum_deployment_os_version": attr.string(mandatory = False),
            "minimum_os_version": attr.string(mandatory = False),
            "platform_type": attr.string(
                default = str(apple_common.platform_type.macos),
            ),
        },
    ),
    fragments = ["apple", "cpp", "objc"],
)
