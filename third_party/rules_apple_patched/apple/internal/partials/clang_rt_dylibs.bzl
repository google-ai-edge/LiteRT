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

"""Partial implementation for Clang runtime libraries processing."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)
load(
    "//apple/internal/utils:clang_rt_dylibs.bzl",
    "clang_rt_dylibs",
)

def _clang_rt_dylibs_partial_impl(
        *,
        actions,
        apple_mac_toolchain_info,
        binary_artifact,
        features,
        label_name,
        output_discriminator,
        platform_prerequisites,
        dylibs):
    """Implementation for the Clang runtime dylibs processing partial."""
    bundle_zips = []
    if clang_rt_dylibs.should_package_clang_runtime(features = features):
        clang_rt_zip = intermediates.file(
            actions = actions,
            target_name = label_name,
            output_discriminator = output_discriminator,
            file_name = "clang_rt.zip",
        )

        clangrttool = apple_mac_toolchain_info.clangrttool
        apple_support.run(
            actions = actions,
            apple_fragment = platform_prerequisites.apple_fragment,
            arguments = [
                binary_artifact.path,
                clang_rt_zip.path,
            ],
            executable = clangrttool,
            # This action needs to read the contents of the Xcode bundle.
            execution_requirements = {"no-sandbox": "1"},
            inputs = [binary_artifact] + dylibs,
            outputs = [clang_rt_zip],
            mnemonic = "ClangRuntimeLibsCopy",
            xcode_config = platform_prerequisites.xcode_version_config,
        )

        bundle_zips.append(
            (processor.location.framework, None, depset([clang_rt_zip])),
        )

    return struct(
        bundle_zips = bundle_zips,
    )

def clang_rt_dylibs_partial(
        *,
        actions,
        apple_mac_toolchain_info,
        binary_artifact,
        dylibs,
        features,
        label_name,
        output_discriminator = None,
        platform_prerequisites):
    """Constructor for the Clang runtime dylibs processing partial.

    Args:
      actions: The actions provider from `ctx.actions`.
      apple_mac_toolchain_info: `struct` of tools from the shared Apple toolchain.
      binary_artifact: The main binary artifact for this target.
      dylibs: List of dylibs (usually from a toolchain).
      features: List of features enabled by the user. Typically from `ctx.features`.
      label_name: Name of the target being built.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      platform_prerequisites: Struct containing information on the platform being targeted.
      dylibs: The clang dylibs to bundle with the target.

    Returns:
      A partial that returns the bundle location of the Clang runtime dylibs, if there were any to
      bundle.
    """
    return partial.make(
        _clang_rt_dylibs_partial_impl,
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        binary_artifact = binary_artifact,
        features = features,
        label_name = label_name,
        output_discriminator = output_discriminator,
        platform_prerequisites = platform_prerequisites,
        dylibs = dylibs,
    )
