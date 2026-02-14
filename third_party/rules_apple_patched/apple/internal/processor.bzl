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

"""Core bundling logic.

The processor module handles the execution of logic for different parts of the
bundling process. This logic is encapsulated into blocks of code called
partials. Each partial will then process a specific aspect of the build process
and will return information on how the bundles should be built.

All partials handled by this processor must follow this API:

  - The expected output is a struct with the following optional fields:
    * bundle_files: Contains tuples of the format
      (location_type, parent_dir, files) where location_type is a field of the
      location enum. The files are then placed at the given location in the
      output bundle.
    * bundle_zips: Contains tuples of the format
      (location_type, parent_dir, files) where location_type is a field of the
      location enum and each file is a ZIP file. The files extracted from the
      ZIPs are then placed at the given location in the output bundle.
    * output_files: Depset of `File`s that should be returned as outputs of the
      target.
    * output_groups: Dictionary of output group names to depset of Files that should be returned in
      the OutputGroupInfo provider.
    * providers: Providers that will be collected and returned by the rule.
    * signed_frameworks: Depset of frameworks which were already signed during the bundling phase.

Location types can be:
  - archive: Files are to be placed relative to the archive of the bundle
    (i.e. the root of the zip/IPA file to generate).
  - app_clip: Files are to be placed in the AppClips section of the bundle.
  - binary: Files are to be placed in the binary section of the bundle.
  - bundle: Files are to be placed at the root of the bundle.
  - content: Files are to be placed in the contents section of the bundle.
  - extension: Files are to be placed in the Extensions section of the bundle.
  - framework: Files are to be placed in the Frameworks section of the bundle.
  - plugin: Files are to be placed in the PlugIns section of the bundle.
  - resources: Files are to be placed in the resources section of the bundle.
  - watch: Files are to be placed inside the Watch section of the bundle. Only applicable for iOS
    apps.

For iOS, tvOS, visionOS, and watchOS, binary, content and resources all refer to the same
location. Only in macOS these paths differ.

All the files given will be symlinked into their expected location in the
bundle, and once complete, the processor will codesign and compress the bundle
into a zip file.

The processor will output a single file, which is the final compressed and
code-signed bundle, and a list of providers that need to be propagated from the
rule.
"""

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
    "//apple/internal:codesigning_support.bzl",
    "codesigning_support",
)
load(
    "//apple/internal:experimental.bzl",
    "is_experimental_tree_artifact_enabled",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:outputs.bzl",
    "outputs",
)
load(
    "//apple/internal/utils:bundle_paths.bzl",
    "bundle_paths",
)
load(
    "//apple/internal/utils:defines.bzl",
    "defines",
)

# Location enum that can be used to tag files into their appropriate location
# in the final archive.
_LOCATION_ENUM = struct(
    app_clip = "app_clip",
    archive = "archive",
    binary = "binary",
    bundle = "bundle",
    content = "content",
    extension = "extension",
    framework = "framework",
    plugin = "plugin",
    resource = "resource",
    watch = "watch",
    xpc_service = "xpc_service",
)

def _invalid_top_level_directories_for_platform(*, platform_type):
    """List of invalid top level directories for the given platform."""

    # As far as we know, there are no locations in macOS bundles that would break
    # codesigning.
    if platform_type == apple_common.platform_type.macos:
        return []

    # Non macOS bundles can't have a top level Resources folder, as it breaks
    # codesigning for some reason. With this, we validate that there are no
    # Resources folder going to be created in the bundle, with a message that
    # better explains which files are incorrectly placed.
    #
    # Since the build may be running on a case-insensitive file-system, which
    # is the default for macOS, this just lists the lowercased ones, so that we
    # can check for all other case variants.
    return ["resources"]

def _is_parent_dir_valid(*, invalid_top_level_dirs, parent_dir):
    """Validates that the files to bundle are not placed in invalid locations.

    codesign will complain when building a non macOS bundle that contains certain
    folders at the top level. We check if there are files that would break
    codesign, and fail early with a nicer message.

    Args:
      invalid_top_level_dirs: String list containing the top level
          directories that have to be avoided when bundling resources.
      parent_dir: String containing the a parent directory inside a bundle.

    Returns:
      False if the parent_dir value is invalid.
    """
    if not parent_dir:
        return True
    for invalid_dir in invalid_top_level_dirs:
        lowercased_parent_dir = parent_dir.lower()
        if lowercased_parent_dir == invalid_dir or lowercased_parent_dir.startswith(invalid_dir + "/"):
            return False
    return True

def _archive_paths(
        *,
        bundle_extension,
        bundle_name,
        embedding = False,
        rule_descriptor,
        tree_artifact_is_enabled):
    """Returns the map of location type to final archive path."""
    if tree_artifact_is_enabled:
        # If experimental tree artifacts are enabled, base all the outputs to be relative to the
        # bundle path.
        bundle_path = ""
    else:
        bundle_name_with_extension = bundle_name + bundle_extension
        if embedding:
            bundle_path = bundle_name_with_extension
        else:
            bundle_path = paths.join(
                rule_descriptor.bundle_locations.archive_relative,
                bundle_name_with_extension,
            )

    contents_path = paths.join(
        bundle_path,
        rule_descriptor.bundle_locations.bundle_relative_contents,
    )

    # Map of location types to relative paths in the archive.
    return {
        _LOCATION_ENUM.app_clip: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_app_clips,
        ),
        _LOCATION_ENUM.archive: "",
        _LOCATION_ENUM.binary: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_binary,
        ),
        _LOCATION_ENUM.bundle: bundle_path,
        _LOCATION_ENUM.content: contents_path,
        _LOCATION_ENUM.extension: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_extensions,
        ),
        _LOCATION_ENUM.framework: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_frameworks,
        ),
        _LOCATION_ENUM.plugin: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_plugins,
        ),
        _LOCATION_ENUM.resource: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_resources,
        ),
        _LOCATION_ENUM.watch: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_watch,
        ),
        _LOCATION_ENUM.xpc_service: paths.join(
            contents_path,
            rule_descriptor.bundle_locations.contents_relative_xpc_service,
        ),
    }

def _bundle_partial_outputs_files(
        *,
        actions,
        apple_mac_toolchain_info,
        apple_xplat_toolchain_info,
        bundle_extension,
        bundle_name,
        codesign_inputs = [],
        codesigning_command = None,
        embedding = False,
        extra_input_files = [],
        ipa_post_processor = None,
        label_name,
        locales_to_include = [],
        output_discriminator,
        output_file,
        partial_outputs,
        platform_prerequisites,
        provisioning_profile,
        rule_descriptor):
    """Invokes bundletool to bundle the files specified by the partial outputs.

    Args:
      actions: The actions provider from `ctx.actions`.
      apple_mac_toolchain_info: A AppleMacToolsToolchainInfo provider.
      apple_xplat_toolchain_info: A AppleXPlatToolsToolchainInfo provider.
      bundle_extension: The extension for the bundle.
      bundle_name: The name of the output bundle.
      codesign_inputs: Extra inputs needed for the `codesign` tool.
      codesigning_command: When building tree artifact outputs, the command to codesign the output
          bundle.
      embedding: Whether outputs are being bundled to be embedded.
      extra_input_files: Extra files to include in the bundling action.
      ipa_post_processor: A file that acts as a bundle post processing tool. May be `None`.
      label_name: The name of the target being built.
      locales_to_include: List of locales to bundle.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      output_file: The file where the final zipped bundle should be created.
      partial_outputs: List of partial outputs from which to collect the files
        that will be bundled inside the final archive.
      platform_prerequisites: Struct containing information on the platform being targeted.
      provisioning_profile: File for the provisioning profile.
      rule_descriptor: A rule descriptor for platform and product types from the rule context.
    """

    # Autotrim locales here only if the rule supports it and there weren't requested locales.
    config_vars = platform_prerequisites.config_vars
    requested_locales_flag = locales_to_include or config_vars.get("apple.locales_to_include")

    trim_locales = defines.bool_value(
        config_vars = config_vars,
        default = None,
        define_name = "apple.trim_lproj_locales",
    ) and rule_descriptor.allows_locale_trimming and requested_locales_flag == None

    control_files = []
    control_zips = []
    input_files = []
    base_locales = ["Base"]

    # Collect the base locales to filter subfolders.
    if trim_locales:
        for partial_output in partial_outputs:
            for _, parent_dir, _ in getattr(partial_output, "bundle_files", []):
                if parent_dir:
                    top_parent = parent_dir.split("/", 1)[0]
                    if top_parent:
                        locale = bundle_paths.locale_for_path(top_parent)
                        if locale:
                            base_locales.append(locale)

    tree_artifact_is_enabled = is_experimental_tree_artifact_enabled(
        platform_prerequisites = platform_prerequisites,
    )

    location_to_paths = _archive_paths(
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        embedding = embedding,
        rule_descriptor = rule_descriptor,
        tree_artifact_is_enabled = tree_artifact_is_enabled,
    )

    platform_type = platform_prerequisites.platform_type
    invalid_top_level_dirs = _invalid_top_level_directories_for_platform(
        platform_type = platform_type,
    )

    processed_file_target_paths = {}
    for partial_output in partial_outputs:
        for location, parent_dir, files in getattr(partial_output, "bundle_files", []):
            if tree_artifact_is_enabled and location == _LOCATION_ENUM.archive:
                # Skip bundling archive related files, as we're only building the bundle directory.
                continue

            if trim_locales:
                locale = bundle_paths.locale_for_path(parent_dir)
                if locale and locale not in base_locales:
                    # Skip files for locales that aren't in the locales for the base resources.
                    continue

            parent_dir_is_valid = _is_parent_dir_valid(
                invalid_top_level_dirs = invalid_top_level_dirs,
                parent_dir = parent_dir,
            )
            if (invalid_top_level_dirs and not parent_dir_is_valid):
                file_paths = "\n".join([f.path for f in files.to_list()])
                fail(("Error: For %s bundles, the following top level " +
                      "directories are invalid (case-insensitive): %s, check input files:\n%s") %
                     (platform_type, ", ".join(invalid_top_level_dirs), file_paths))

            sources = files.to_list()
            input_files.extend(sources)

            for source in sources:
                target_path = paths.join(location_to_paths[location], parent_dir or "")

                if not source.is_directory:
                    target_path = paths.join(target_path, source.basename)
                    if target_path in processed_file_target_paths:
                        fail(
                            ("Multiple files would be placed at \"%s\" in the bundle, which " +
                             "is not allowed. check input file:\n%s") % (target_path, source.path),
                        )
                    processed_file_target_paths[target_path] = None
                control_files.append(struct(src = source.path, dest = target_path))

        for location, parent_dir, zip_files in getattr(partial_output, "bundle_zips", []):
            if tree_artifact_is_enabled and location == _LOCATION_ENUM.archive:
                # Skip bundling archive related files, as we're only building the bundle directory.
                continue

            parent_dir_is_valid = _is_parent_dir_valid(
                invalid_top_level_dirs = invalid_top_level_dirs,
                parent_dir = parent_dir,
            )
            if invalid_top_level_dirs and not parent_dir_is_valid:
                fail(("Error: For %s bundles, the following top level " +
                      "directories are invalid (case-insensitive): %s, check input files:\n%s") %
                     (platform_type, ", ".join(invalid_top_level_dirs)))

            sources = zip_files.to_list()
            input_files.extend(sources)

            for source in sources:
                target_path = paths.join(location_to_paths[location], parent_dir or "")
                control_zips.append(struct(src = source.path, dest = target_path))

    post_processor = ipa_post_processor
    post_processor_path = ""

    if post_processor:
        post_processor_path = post_processor.path

    control = struct(
        bundle_merge_files = control_files,
        bundle_merge_zips = control_zips,
        output = output_file.path,
        code_signing_commands = codesigning_command or "",
        post_processor = post_processor_path,
    )

    if embedding:
        control_file_name = "embedding_bundletool_control.json"
    else:
        control_file_name = "bundletool_control.json"

    control_file = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = control_file_name,
    )
    actions.write(
        output = control_file,
        content = json.encode(control),
    )

    bundletool_inputs = input_files + [control_file] + extra_input_files

    action_args = {
        "arguments": [control_file.path],
        "outputs": [output_file],
    }

    if tree_artifact_is_enabled:
        # Since the tree artifact bundler also runs the post processor and codesigning, this
        # action needs to run on a macOS machine.

        bundletool = apple_mac_toolchain_info.bundletool_experimental

        # Required to satisfy an implicit dependency, when the codesigning commands are executed by
        # the experimental bundle tool script.
        codesigningtool = apple_mac_toolchain_info.codesigningtool

        bundling_tools = [bundletool, codesigningtool]
        if post_processor:
            bundling_tools.append(post_processor)

        execution_requirements = {
            # Unsure, but may be needed for keychain access, especially for
            # files that live in $HOME.
            "no-sandbox": "1",
        }

        if (codesigning_command and
            platform_prerequisites.platform.is_device and
            provisioning_profile):
            # Added so that the output of this action is not cached remotely,
            # in case multiple developers sign the same artifact with different
            # identities.
            execution_requirements["no-remote"] = "1"

        apple_support.run(
            actions = actions,
            apple_fragment = platform_prerequisites.apple_fragment,
            executable = bundletool,
            execution_requirements = execution_requirements,
            inputs = bundletool_inputs + codesign_inputs,
            mnemonic = "BundleTreeApp",
            progress_message = "Bundling, processing and signing %s" % label_name,
            tools = bundling_tools,
            xcode_config = platform_prerequisites.xcode_version_config,
            **action_args
        )
    else:
        bundletool = apple_xplat_toolchain_info.bundletool
        actions.run(
            executable = bundletool,
            inputs = bundletool_inputs,
            mnemonic = "BundleApp",
            progress_message = "Bundling %s" % label_name,
            **action_args
        )

def _bundle_post_process_and_sign(
        *,
        actions,
        apple_mac_toolchain_info,
        apple_xplat_toolchain_info,
        bundle_extension,
        bundle_name,
        codesign_inputs,
        codesignopts,
        entitlements,
        features,
        ipa_post_processor,
        locales_to_include,
        output_archive,
        output_discriminator,
        partial_outputs,
        platform_prerequisites,
        predeclared_outputs,
        process_and_sign_template,
        provisioning_profile,
        rule_descriptor,
        rule_label):
    """Bundles, post-processes and signs the files in partial_outputs.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_mac_toolchain_info: A AppleMacToolsToolchainInfo provider.
        apple_xplat_toolchain_info: A AppleXPlatToolsToolchainInfo provider.
        bundle_extension: The extension for the bundle.
        bundle_name: The name of the output bundle.
        codesign_inputs: Extra inputs needed for the `codesign` tool.
        codesignopts: Extra options to pass to the `codesign` tool.
        entitlements: The entitlements file to sign with. Can be `None` if one was not provided.
        features: List of features enabled by the user. Typically from `ctx.features`.
        ipa_post_processor: A file that acts as a bundle post processing tool. May be `None`.
        locales_to_include: List of locales to bundle.
        output_archive: The file representing the final bundled, post-processed and signed archive.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        partial_outputs: The outputs of the partials used to process this target's bundle.
        platform_prerequisites: Struct containing information on the platform being targeted.
        predeclared_outputs: Outputs declared by the owning context. Typically from `ctx.outputs`.
        process_and_sign_template: A template for a shell script to process and sign as a file.
        provisioning_profile: File for the provisioning profile.
        rule_descriptor: A rule descriptor for platform and product types from the rule context.
        rule_label: The label of the target being analyzed.
    """
    tree_artifact_is_enabled = is_experimental_tree_artifact_enabled(
        platform_prerequisites = platform_prerequisites,
    )
    archive_paths = _archive_paths(
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        rule_descriptor = rule_descriptor,
        tree_artifact_is_enabled = tree_artifact_is_enabled,
    )
    signed_frameworks_depsets = []
    for partial_output in partial_outputs:
        if hasattr(partial_output, "signed_frameworks"):
            signed_frameworks_depsets.append(partial_output.signed_frameworks)
    transitive_signed_frameworks = depset(transitive = signed_frameworks_depsets)

    if tree_artifact_is_enabled:
        extra_input_files = []

        if entitlements:
            extra_input_files.append(entitlements)

        if provisioning_profile:
            extra_input_files.append(provisioning_profile)

        # TODO(b/149874635): Don't pass frameworks_path unless the rule has it (*_application).
        codesigning_command = codesigning_support.codesigning_command(
            codesigningtool = apple_mac_toolchain_info.codesigningtool.executable,
            entitlements = entitlements,
            features = features,
            frameworks_path = archive_paths[_LOCATION_ENUM.framework],
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
            signed_frameworks = transitive_signed_frameworks,
            codesignopts = codesignopts,
        )

        _bundle_partial_outputs_files(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            codesign_inputs = codesign_inputs,
            codesigning_command = codesigning_command,
            extra_input_files = extra_input_files,
            ipa_post_processor = ipa_post_processor,
            label_name = rule_label.name,
            locales_to_include = locales_to_include,
            output_discriminator = output_discriminator,
            output_file = output_archive,
            partial_outputs = partial_outputs,
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        )

        actions.write(
            output = predeclared_outputs.archive,
            content = "This is dummy file because tree artifacts are enabled",
        )
    else:
        # This output, is an intermediate artifact used for post processing, signing, etc.
        unprocessed_archive = intermediates.file(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            file_name = "unprocessed_archive.zip",
        )
        _bundle_partial_outputs_files(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            ipa_post_processor = ipa_post_processor,
            label_name = rule_label.name,
            locales_to_include = locales_to_include,
            output_discriminator = output_discriminator,
            output_file = unprocessed_archive,
            partial_outputs = partial_outputs,
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        )

        archive_codesigning_path = archive_paths[_LOCATION_ENUM.bundle]
        frameworks_path = archive_paths[_LOCATION_ENUM.framework]

        output_archive_root_path = outputs.root_path_from_archive(archive = output_archive)

        # TODO(b/149874635): Don't pass frameworks_path unless the rule has it (ios_application).
        codesigning_support.post_process_and_sign_archive_action(
            actions = actions,
            archive_codesigning_path = archive_codesigning_path,
            codesign_inputs = codesign_inputs,
            codesigningtool = apple_mac_toolchain_info.codesigningtool,
            codesignopts = codesignopts,
            entitlements = entitlements,
            features = features,
            frameworks_path = frameworks_path,
            input_archive = unprocessed_archive,
            ipa_post_processor = ipa_post_processor,
            label_name = rule_label.name,
            output_archive = output_archive,
            output_archive_root_path = output_archive_root_path,
            output_discriminator = output_discriminator,
            platform_prerequisites = platform_prerequisites,
            process_and_sign_template = process_and_sign_template,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
            signed_frameworks = transitive_signed_frameworks,
        )

        has_different_embedding_archive = outputs.has_different_embedding_archive(
            platform_prerequisites = platform_prerequisites,
            rule_descriptor = rule_descriptor,
        )
        if has_different_embedding_archive:
            embedding_archive = outputs.archive_for_embedding(
                actions = actions,
                bundle_extension = bundle_extension,
                bundle_name = bundle_name,
                label_name = rule_label.name,
                rule_descriptor = rule_descriptor,
                platform_prerequisites = platform_prerequisites,
                predeclared_outputs = predeclared_outputs,
            )
            embedding_archive_paths = _archive_paths(
                bundle_extension = bundle_extension,
                bundle_name = bundle_name,
                embedding = True,
                rule_descriptor = rule_descriptor,
                tree_artifact_is_enabled = tree_artifact_is_enabled,
            )
            embedding_archive_codesigning_path = embedding_archive_paths[_LOCATION_ENUM.bundle]
            embedding_frameworks_path = embedding_archive_paths[_LOCATION_ENUM.framework]
            embedding_archive_root_path = outputs.root_path_from_archive(archive = embedding_archive)
            unprocessed_embedded_archive = intermediates.file(
                actions = actions,
                target_name = rule_label.name,
                output_discriminator = output_discriminator,
                file_name = "unprocessed_embedded_archive.zip",
            )
            _bundle_partial_outputs_files(
                actions = actions,
                apple_mac_toolchain_info = apple_mac_toolchain_info,
                apple_xplat_toolchain_info = apple_xplat_toolchain_info,
                bundle_extension = bundle_extension,
                bundle_name = bundle_name,
                embedding = True,
                ipa_post_processor = ipa_post_processor,
                label_name = rule_label.name,
                locales_to_include = locales_to_include,
                output_discriminator = output_discriminator,
                output_file = unprocessed_embedded_archive,
                partial_outputs = partial_outputs,
                platform_prerequisites = platform_prerequisites,
                provisioning_profile = provisioning_profile,
                rule_descriptor = rule_descriptor,
            )

            codesigning_support.post_process_and_sign_archive_action(
                actions = actions,
                archive_codesigning_path = embedding_archive_codesigning_path,
                codesign_inputs = codesign_inputs,
                codesigningtool = apple_mac_toolchain_info.codesigningtool,
                codesignopts = codesignopts,
                entitlements = entitlements,
                features = features,
                frameworks_path = embedding_frameworks_path,
                input_archive = unprocessed_embedded_archive,
                ipa_post_processor = ipa_post_processor,
                label_name = rule_label.name,
                output_archive = embedding_archive,
                output_archive_root_path = embedding_archive_root_path,
                output_discriminator = output_discriminator,
                platform_prerequisites = platform_prerequisites,
                process_and_sign_template = process_and_sign_template,
                provisioning_profile = provisioning_profile,
                rule_descriptor = rule_descriptor,
                signed_frameworks = transitive_signed_frameworks,
            )

def _process(
        *,
        actions,
        apple_mac_toolchain_info,
        apple_xplat_toolchain_info,
        bundle_extension,
        bundle_name,
        bundle_post_process_and_sign = True,
        codesign_inputs = [],
        codesignopts = [],
        entitlements = None,
        features,
        ipa_post_processor = None,
        locales_to_include = [],
        output_discriminator = None,
        partials,
        platform_prerequisites,
        predeclared_outputs,
        process_and_sign_template,
        provisioning_profile = None,
        rule_descriptor,
        rule_label):
    """Processes a list of partials that provide the files to be bundled.

    Args:
      actions: The actions provider from `ctx.actions`.
      apple_mac_toolchain_info: A AppleMacToolsToolchainInfo provider.
      apple_xplat_toolchain_info: A AppleXPlatToolsToolchainInfo provider.
      bundle_extension: The extension for the bundle.
      bundle_name: The name of the output bundle.
      bundle_post_process_and_sign: If the process action should also post process and sign after
          calling the implementation of every partial. Defaults to True.
      codesign_inputs: Extra inputs needed for the `codesign` tool.
      codesignopts: Extra options to pass to the `codesign` tool.
      entitlements: The entitlements file to sign with. Can be `None` if one was not provided.
      features: List of features enabled by the user. Typically from `ctx.features`.
      ipa_post_processor: A file that acts as a bundle post processing tool. Defaults to `None`.
      locales_to_include: List of locales to explicitly include in the bundle. Defaults tp `[]`.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      partials: The list of partials to process to construct the complete bundle.
      platform_prerequisites: Struct containing information on the platform being targeted.
      predeclared_outputs: Outputs declared by the owning context. Typically from `ctx.outputs`.
      process_and_sign_template: A template for a shell script to process and sign as a file.
      provisioning_profile: File for the provisioning profile.
      rule_descriptor: A rule descriptor for platform and product types from the rule context.
      rule_label: The label of the target being analyzed.

    Returns:
      A `struct` with the results of the processing. The files to make outputs of
      the rule are contained under the `output_files` field, the providers to
      return are contained under the `providers` field, and a dictionary containing
      any additional output groups is in the `output_groups` field.
    """

    partial_outputs = [partial.call(p) for p in partials]

    if bundle_post_process_and_sign:
        output_archive = outputs.archive(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            label_name = rule_label.name,
            output_discriminator = output_discriminator,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            rule_descriptor = rule_descriptor,
        )
        _bundle_post_process_and_sign(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            codesign_inputs = codesign_inputs,
            codesignopts = codesignopts,
            entitlements = entitlements,
            features = features,
            ipa_post_processor = ipa_post_processor,
            locales_to_include = locales_to_include,
            output_archive = output_archive,
            output_discriminator = output_discriminator,
            partial_outputs = partial_outputs,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            process_and_sign_template = process_and_sign_template,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
            rule_label = rule_label,
        )
        transitive_output_files = [depset([output_archive])]
    else:
        transitive_output_files = []

    providers = []
    output_group_dicts = []
    for partial_output in partial_outputs:
        if hasattr(partial_output, "providers"):
            providers.extend(partial_output.providers)
        if hasattr(partial_output, "output_files"):
            transitive_output_files.append(partial_output.output_files)
        if hasattr(partial_output, "output_groups"):
            output_group_dicts.append(partial_output.output_groups)

    return struct(
        output_files = depset(transitive = transitive_output_files),
        output_groups = outputs.merge_output_groups(*output_group_dicts),
        providers = providers,
    )

processor = struct(
    process = _process,
    location = _LOCATION_ENUM,
)
