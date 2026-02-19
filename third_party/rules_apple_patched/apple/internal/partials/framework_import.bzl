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

"""Partial implementation for framework import file processing."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple:providers.bzl",
    "AppleFrameworkImportInfo",
)
load(
    "//apple/internal:codesigning_support.bzl",
    "codesigning_support",
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
    "//apple/internal/utils:bundle_paths.bzl",
    "bundle_paths",
)

def _framework_import_partial_impl(
        *,
        actions,
        apple_mac_toolchain_info,
        features,
        label_name,
        output_discriminator,
        platform_prerequisites,
        provisioning_profile,
        rule_descriptor,
        targets,
        targets_to_avoid):
    """Implementation for the framework import file processing partial."""
    transitive_sets = [
        x[AppleFrameworkImportInfo].framework_imports
        for x in targets
        if AppleFrameworkImportInfo in x and
           hasattr(x[AppleFrameworkImportInfo], "framework_imports")
    ]
    files_to_bundle = depset(transitive = transitive_sets).to_list()

    if targets_to_avoid:
        avoid_transitive_sets = [
            x[AppleFrameworkImportInfo].framework_imports
            for x in targets_to_avoid
            if AppleFrameworkImportInfo in x and
               hasattr(x[AppleFrameworkImportInfo], "framework_imports")
        ]
        if avoid_transitive_sets:
            avoid_files = depset(transitive = avoid_transitive_sets).to_list()

            # Remove any files present in the targets to avoid from framework files that need to be
            # bundled.
            files_to_bundle = [x for x in files_to_bundle if x not in avoid_files]

    # Collect the architectures that we are using for the build.
    build_archs_found = depset(transitive = [
        x[AppleFrameworkImportInfo].build_archs
        for x in targets
        if AppleFrameworkImportInfo in x
    ]).to_list()

    # Start assembling our partial's outputs.
    bundle_zips = []
    signed_frameworks_list = []

    # Separating our files by framework path, to better address what should be passed in.
    framework_binaries_by_framework = dict()
    files_by_framework = dict()

    for file in files_to_bundle:
        framework_path = bundle_paths.farthest_parent(file.short_path, "framework")

        # Use the framework path's basename to distinguish groups of files.
        framework_basename = paths.basename(framework_path)
        if not files_by_framework.get(framework_basename):
            files_by_framework[framework_basename] = []
        if not framework_binaries_by_framework.get(framework_basename):
            framework_binaries_by_framework[framework_basename] = []

        # Check if file is a tree artifact to treat as bundle files.
        # XCFramework import rules forward tree artifacts when using the
        # xcframework_processor_tool, since the effective XCFramework library
        # files are not known during analysis phase.
        if file.is_directory:
            files_by_framework[framework_basename].append(file)
            continue

        # Check if this file is a binary to slice and code sign.
        framework_relative_path = paths.relativize(file.short_path, framework_path)

        parent_dir = framework_basename
        framework_relative_dir = paths.dirname(framework_relative_path).strip("/")
        if framework_relative_dir:
            parent_dir = paths.join(parent_dir, framework_relative_dir)

        # Classify if it's a file to bundle or framework binary
        if paths.replace_extension(parent_dir, "") == file.basename:
            framework_binaries_by_framework[framework_basename].append(file)
        else:
            files_by_framework[framework_basename].append(file)

    for framework_basename in files_by_framework.keys():
        # Create a temporary path for intermediate files and the anticipated zip output.
        temp_path = paths.join("_imported_frameworks/", framework_basename)
        framework_zip = intermediates.file(
            actions = actions,
            target_name = label_name,
            output_discriminator = output_discriminator,
            file_name = temp_path + ".zip",
        )
        temp_framework_bundle_path = paths.split_extension(framework_zip.path)[0]

        # Pass through all binaries, files, and relevant info as args.
        args = actions.args()

        args.add_all(
            framework_binaries_by_framework[framework_basename],
            before_each = "--framework_binary",
        )

        args.add_all(build_archs_found, before_each = "--slice")

        args.add("--strip_bitcode")

        args.add("--output_zip", framework_zip.path)

        args.add("--temp_path", temp_framework_bundle_path)

        args.add_all(files_by_framework[framework_basename], before_each = "--framework_file")

        codesign_args = codesigning_support.codesigning_args(
            entitlements = None,
            features = features,
            full_archive_path = temp_framework_bundle_path,
            is_framework = True,
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        )
        if codesign_args:
            args.add_all(codesign_args)
        else:
            # Add required argument to disable signing because
            # code sign arguments are mutually exclusive groups.
            args.add("--disable_signing")

        codesigningtool = apple_mac_toolchain_info.codesigningtool
        imported_dynamic_framework_processor = apple_mac_toolchain_info.imported_dynamic_framework_processor

        execution_requirements = {}

        # Inputs of action are all the framework files, plus binaries needed for identifying the
        # current build's preferred architecture, and the provisioning profile if specified.
        input_files = (
            files_by_framework[framework_basename] +
            framework_binaries_by_framework[framework_basename]
        )
        if codesign_args and provisioning_profile:
            input_files.append(provisioning_profile)
            execution_requirements = {"no-sandbox": "1"}
            if platform_prerequisites.platform.is_device:
                # Added so that the output of this action is not cached
                # remotely, in case multiple developers sign the same artifact
                # with different identities.
                execution_requirements["no-remote"] = "1"

        apple_support.run(
            actions = actions,
            apple_fragment = platform_prerequisites.apple_fragment,
            arguments = [args],
            executable = imported_dynamic_framework_processor,
            execution_requirements = execution_requirements,
            inputs = input_files,
            mnemonic = "ImportedDynamicFrameworkProcessor",
            outputs = [framework_zip],
            tools = [codesigningtool],
            xcode_config = platform_prerequisites.xcode_version_config,
        )

        bundle_zips.append(
            (processor.location.framework, None, depset([framework_zip])),
        )
        signed_frameworks_list.append(framework_basename)

    return struct(
        bundle_zips = bundle_zips,
        signed_frameworks = depset(signed_frameworks_list),
    )

def framework_import_partial(
        *,
        actions,
        apple_mac_toolchain_info,
        features,
        label_name,
        output_discriminator = None,
        platform_prerequisites,
        provisioning_profile,
        rule_descriptor,
        targets,
        targets_to_avoid = []):
    """Constructor for the framework import file processing partial.

    This partial propagates framework import file bundle locations. The files are collected through
    the framework_provider_aspect aspect.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_mac_toolchain_info: `struct` of tools from the shared Apple toolchain.
        features: List of features enabled by the user. Typically from `ctx.features`.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        platform_prerequisites: Struct containing information on the platform being targeted.
        provisioning_profile: File for the provisioning profile.
        rule_descriptor: A rule descriptor for platform and product types from the rule context.
        targets: The list of targets through which to collect the framework import files.
        targets_to_avoid: The list of targets that may already be bundling some of the frameworks,
            to be used when deduplicating frameworks already bundled.

    Returns:
        A partial that returns the bundle location of the framework import files.
    """
    return partial.make(
        _framework_import_partial_impl,
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        features = features,
        label_name = label_name,
        output_discriminator = output_discriminator,
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        targets = targets,
        targets_to_avoid = targets_to_avoid,
    )
