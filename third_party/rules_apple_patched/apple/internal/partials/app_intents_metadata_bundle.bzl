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

"""Partial implementation for processing AppIntents metadata bundle."""

load("@bazel_skylib//lib:partial.bzl", "partial")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("//apple/internal:intermediates.bzl", "intermediates")
load("//apple/internal:linking_support.bzl", "linking_support")
load("//apple/internal:processor.bzl", "processor")
load(
    "//apple/internal/providers:app_intents_info.bzl",
    "AppIntentsInfo",
)
load(
    "//apple/internal/resource_actions:app_intents.bzl",
    "generate_app_intents_metadata_bundle",
)

def _app_intents_metadata_bundle_partial_impl(
        *,
        actions,
        cc_toolchains,
        ctx,
        deps,
        disabled_features,
        features,
        label,
        platform_prerequisites):
    """Implementation of the AppIntents metadata bundle partial."""
    if not deps:
        # No `app_intents` were set by the rule calling this partial.
        return struct()

    # Link 'stub' binary to use for app intents metadata processing.
    # This binary should only contain symbols for structs implementing the AppIntents protocol.
    # Instead of containing all the application/extension/framework binary symbols, allowing
    # the action to run faster and avoid depending on the application binary linking step.
    #
    # TODO(b/295227222): Avoid this linker step for Xcode 15.0+ when rules_swift supports the new
    # swiftconstvalues-based manner of handling App Intents metadata.
    link_result = linking_support.legacy_link_multi_arch_binary(
        actions = actions,
        cc_toolchains = cc_toolchains,
        ctx = ctx,
        deps = deps,
        disabled_features = disabled_features,
        features = features,
        label = label,
        user_link_flags = [
            # Force _NSExtensionMain, which exists on all Apple platforms, to
            # be the main symbol for the binary, just so any main symbol will
            # exist. Since this binary is discarded afterwards the main symbol
            # doesn't actually matter. This can be removed when the TODO above
            # is resolved.
            "-Wl,-e,_NSExtensionMain",
            # Force the binary to link Foundation to make the hack above work.
            "-Wl,-framework,Foundation",
        ],
    )

    fat_stub_binary = intermediates.file(
        actions = actions,
        target_name = label.name,
        output_discriminator = None,
        file_name = "{}_app_intents_stub_binary".format(label.name),
    )

    linking_support.lipo_or_symlink_inputs(
        actions = actions,
        inputs = [output.binary for output in link_result.outputs],
        output = fat_stub_binary,
        apple_fragment = platform_prerequisites.apple_fragment,
        xcode_config = platform_prerequisites.xcode_version_config,
    )

    label.relative(
        label.name + "_app_intents_stub_binary",
    )

    # Mirroring Xcode 15+ behavior, the metadata tool only looks at the first split for a given arch
    # rather than every possible set of source files and inputs. Oddly, this only applies to the
    # swift source files and the swiftconstvalues files; the triples and other files do cover all
    # available archs.
    first_cc_toolchain_key = cc_toolchains.keys()[0]

    metadata_bundle = generate_app_intents_metadata_bundle(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        bundle_binary = fat_stub_binary,
        constvalues_files = [
            swiftconstvalues_file
            for dep in deps[first_cc_toolchain_key]
            for swiftconstvalues_file in dep[AppIntentsInfo].swiftconstvalues_files
        ],
        intents_module_names = [
            intent_module_name
            for dep in deps[first_cc_toolchain_key]
            for intent_module_name in dep[AppIntentsInfo].intent_module_names
        ],
        label = label,
        source_files = [
            swift_source_file
            for dep in deps[first_cc_toolchain_key]
            for swift_source_file in dep[AppIntentsInfo].swift_source_files
        ],
        target_triples = [
            cc_toolchain[cc_common.CcToolchainInfo].target_gnu_system_name
            for cc_toolchain in cc_toolchains.values()
        ],
        xcode_version_config = platform_prerequisites.xcode_version_config,
    )

    bundle_location = processor.location.bundle
    if str(platform_prerequisites.platform_type) == "macos":
        bundle_location = processor.location.resource

    return struct(
        bundle_files = [(
            bundle_location,
            "Metadata.appintents",
            depset(direct = [metadata_bundle]),
        )],
    )

def app_intents_metadata_bundle_partial(
        *,
        actions,
        cc_toolchains,
        ctx,
        deps,
        disabled_features,
        features,
        label,
        platform_prerequisites):
    """Constructor for the AppIntents metadata bundle processing partial.

    This partial generates the Metadata.appintents bundle required for AppIntents functionality.

    Args:
        actions: The actions provider from ctx.actions.
        cc_toolchains: Dictionary of CcToolchainInfo and ApplePlatformInfo providers under a split
            transition to relay target platform information.
        ctx: The Starlark context for a rule target being built.
        deps: Dictionary of targets under a split transition implementing the AppIntents protocol.
        disabled_features: List of features to be disabled for C++ link actions.
        features: List of features to be enabled for C++ link actions.
        label: Label of the target being built.
        platform_prerequisites: Struct containing information on the platform being targeted.
    Returns:
        A partial that generates the Metadata.appintents bundle.
    """
    return partial.make(
        _app_intents_metadata_bundle_partial_impl,
        actions = actions,
        cc_toolchains = cc_toolchains,
        ctx = ctx,
        deps = deps,
        disabled_features = disabled_features,
        features = features,
        label = label,
        platform_prerequisites = platform_prerequisites,
    )
