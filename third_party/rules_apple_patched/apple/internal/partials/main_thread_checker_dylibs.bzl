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

"""Partial implementation for Main Thread Checker libraries processing."""

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
    "//apple/internal/utils:main_thread_checker_dylibs.bzl",
    "main_thread_checker_dylibs",
)

def _should_include_main_thread_checker(features):
    return main_thread_checker_dylibs.should_package_main_thread_checker_dylib(features = features)

def _create_main_thread_checker_dylib(actions, label_name, output_discriminator):
    return intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "libMainThreadChecker.dylib",
    )

def _run_main_thread_checker(
        actions,
        binary_artifact,
        dylibs,
        main_thread_checker_dylib,
        main_thread_checker_tool,
        platform_prerequisites):
    apple_support.run(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        arguments = [main_thread_checker_dylib.path],
        executable = main_thread_checker_tool,
        execution_requirements = {"no-sandbox": "1"},
        inputs = [binary_artifact] + dylibs,
        outputs = [main_thread_checker_dylib],
        mnemonic = "MainThreadCheckerLibsCopy",
        xcode_config = platform_prerequisites.xcode_version_config,
    )

def _main_thread_checker_dylibs_partial_impl(
        *,
        actions,
        apple_mac_toolchain_info,
        binary_artifact,
        features,
        label_name,
        output_discriminator,
        platform_prerequisites,
        dylibs):
    """Implementation for the Main Thread Checker dylibs processing partial."""
    bundle_files = []

    if not _should_include_main_thread_checker(features = features):
        return struct(bundle_files = bundle_files)

    main_thread_checker_dylib = _create_main_thread_checker_dylib(actions, label_name, output_discriminator)
    main_thread_checker_tool = apple_mac_toolchain_info.main_thread_checker_tool

    _run_main_thread_checker(actions, binary_artifact, dylibs, main_thread_checker_dylib, main_thread_checker_tool, platform_prerequisites)

    bundle_files.append(
        (processor.location.framework, None, depset([main_thread_checker_dylib])),
    )

    return struct(bundle_files = bundle_files)

def main_thread_checker_dylibs_partial(
        *,
        actions,
        apple_mac_toolchain_info,
        binary_artifact,
        dylibs,
        features,
        label_name,
        output_discriminator = None,
        platform_prerequisites):
    """Constructor for the Main Thread Checker dylibs processing partial.

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
      dylibs: The Main Thread Checker dylibs to bundle with the target.

    Returns:
      A partial that returns the bundle location of the Main Thread Checker dylib, if there were any to
      bundle.
    """
    return partial.make(
        _main_thread_checker_dylibs_partial_impl,
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        binary_artifact = binary_artifact,
        features = features,
        label_name = label_name,
        output_discriminator = output_discriminator,
        platform_prerequisites = platform_prerequisites,
        dylibs = dylibs,
    )
