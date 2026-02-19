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
# limitations under the License.

"""Partial implementation for the AppleBundleInfo provider."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "//apple/internal:outputs.bzl",
    "outputs",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebundleinfo",
)

def _apple_bundle_info_partial_impl(
        *,
        actions,
        bundle_id,
        bundle_extension,
        bundle_name,
        entitlements,
        executable_name,
        extension_safe,
        label_name,
        output_discriminator,
        platform_prerequisites,
        predeclared_outputs,
        product_type,
        rule_descriptor):
    """Implementation for the AppleBundleInfo processing partial."""

    archive = outputs.archive(
        actions = actions,
        bundle_name = bundle_name,
        bundle_extension = bundle_extension,
        label_name = label_name,
        output_discriminator = output_discriminator,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )
    archive_root = outputs.root_path_from_archive(archive = archive)

    binary = outputs.binary(
        actions = actions,
        bundle_name = bundle_name,
        executable_name = executable_name,
        label_name = label_name,
        output_discriminator = output_discriminator,
    )

    infoplist = None
    if bundle_id:
        # Only add the infoplist if there is a bundle ID, otherwise, do not create the output file.
        infoplist = outputs.infoplist(
            actions = actions,
            label_name = label_name,
            output_discriminator = output_discriminator,
        )

    return struct(
        providers = [
            new_applebundleinfo(
                archive = archive,
                archive_root = archive_root,
                binary = binary,
                bundle_id = bundle_id,
                bundle_name = bundle_name,
                bundle_extension = bundle_extension,
                entitlements = entitlements,
                executable_name = executable_name,
                extension_safe = extension_safe,
                infoplist = infoplist,
                minimum_deployment_os_version = platform_prerequisites.minimum_deployment_os,
                minimum_os_version = platform_prerequisites.minimum_os,
                platform_type = str(platform_prerequisites.platform_type),
                product_type = product_type,
                uses_swift = platform_prerequisites.uses_swift,
            ),
        ],
    )

def apple_bundle_info_partial(
        *,
        actions,
        bundle_id = None,
        bundle_extension,
        bundle_name,
        executable_name,
        extension_safe = False,
        entitlements = None,
        label_name,
        output_discriminator = None,
        platform_prerequisites,
        predeclared_outputs,
        product_type,
        rule_descriptor):
    """Constructor for the AppleBundleInfo processing partial.

    This partial propagates the AppleBundleInfo provider for this target.

    Args:
      actions: The actions provider from ctx.actions.
      bundle_id: The bundle ID to configure for this target.
      bundle_extension: Extension for the Apple bundle inside the archive.
      bundle_name: The name of the output bundle.
      entitlements: The entitlements file to sign with. Can be `None` if one was not provided.
      executable_name: The name of the output executable.
      extension_safe: True if the target propagating this provider was compiled
        and linked with -application-extension, restricting it to extension-safe
        APIs only.
      label_name: Name of the target being built.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      platform_prerequisites: Struct containing information on the platform being targeted.
      predeclared_outputs: Outputs declared by the owning context. Typically from `ctx.outputs`.
      product_type: Product type identifier used to describe the current bundle type.
      rule_descriptor: The rule descriptor for the given rule.

    Returns:
      A partial that returns the AppleBundleInfo provider.
    """
    return partial.make(
        _apple_bundle_info_partial_impl,
        actions = actions,
        bundle_id = bundle_id,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        executable_name = executable_name or bundle_name,
        extension_safe = extension_safe,
        entitlements = entitlements,
        label_name = label_name,
        output_discriminator = output_discriminator,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        product_type = product_type,
        rule_descriptor = rule_descriptor,
    )
