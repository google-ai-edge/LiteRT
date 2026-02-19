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

"""Plist related actions."""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@bazel_skylib//lib:shell.bzl",
    "shell",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple:providers.bzl",
    "AppleBundleVersionInfo",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:platform_support.bzl",
    "platform_support",
)

def plisttool_action(
        *,
        actions,
        control_file,
        inputs,
        mnemonic = None,
        outputs,
        platform_prerequisites,
        plisttool):
    """Registers an action that invokes `plisttool`.

    This function is a low-level helper that simply invokes `plisttool` with the given arguments.
    It is intended to be called by other functions that register actions for more specific
    resources, like Info.plist files or entitlements.

    Args:
      actions: The actions provider from `ctx.actions`.
      control_file: The `File` containing the control struct to be passed to plisttool.
      inputs: Any `File`s that should be treated as inputs to the underlying action.
      mnemonic: The mnemonic to display when the action executes. Defaults to None.
      outputs: Any `File`s that should be treated as outputs of the underlying action.
      platform_prerequisites: Struct containing information on the platform being targeted.
      plisttool: A files_to_run for the plist tool.
    """
    apple_support.run(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        arguments = [control_file.path],
        executable = plisttool,
        inputs = inputs + [control_file],
        mnemonic = mnemonic,
        outputs = outputs,
        xcode_config = platform_prerequisites.xcode_version_config,
    )

def compile_plist(*, actions, input_file, output_file, platform_prerequisites):
    """Creates an action that compiles plist and strings files.

    Args:
      actions: The actions provider from `ctx.actions`.
      input_file: The property list file that should be converted.
      output_file: The file reference for the output plist.
      platform_prerequisites: Struct containing information on the platform being targeted.
    """
    if input_file.basename.endswith(".strings"):
        mnemonic = "CompileStrings"
    else:
        mnemonic = "CompilePlist"

    # This command will check whether the input file is non-empty, and then
    # execute the version of plutil that takes the file directly. If the file is
    # empty, it will echo an new line and then pipe it into plutil. We do this
    # to handle empty files as plutil doesn't handle them very well.
    plutil_command = "plutil -convert binary1 -o %s --" % shell.quote(output_file.path)
    complete_command = ("if [[ -s {in_file} ]] ; then {plutil_command} {in_file} ; " +
                        "elif [[ -f {in_file} ]] ; then echo | {plutil_command} - ; " +
                        "else exit 1 ; " +
                        "fi").format(
        in_file = shell.quote(input_file.path),
        plutil_command = plutil_command,
    )
    apple_support.run_shell(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        command = complete_command,
        inputs = [input_file],
        mnemonic = mnemonic,
        outputs = [output_file],
        xcode_config = platform_prerequisites.xcode_version_config,
    )

def merge_resource_infoplists(
        *,
        actions,
        bundle_id,
        bundle_name_with_extension,
        input_files,
        output_discriminator,
        output_plist,
        platform_prerequisites,
        plisttool,
        rule_label):
    """Merges a list of plist files for resource bundles with substitutions.

    Args:
      actions: The actions provider from `ctx.actions`.
      bundle_id: The bundle ID to use when templating plist files.
      bundle_name_with_extension: The full name of the bundle where the plist will be placed.
      input_files: The list of plists to merge.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      output_plist: The file reference for the output plist.
      platform_prerequisites: Struct containing information on the platform being targeted.
      plisttool: A files_to_run for the plist tool.
      rule_label: The label of the target being analyzed.
    """
    product_name = paths.replace_extension(bundle_name_with_extension, "")
    substitutions = {
        "BUNDLE_NAME": bundle_name_with_extension,
        "PRODUCT_NAME": product_name,
        "TARGET_NAME": product_name,
    }
    if bundle_id:
        substitutions["PRODUCT_BUNDLE_IDENTIFIER"] = bundle_id

    # The generated Info.plists from Xcode's project templates use
    # DEVELOPMENT_LANGUAGE as the default variable substitution for
    # CFBundleDevelopmentRegion. We substitute this to `en` to support
    # Info.plists out of the box coming from Xcode.
    substitutions["DEVELOPMENT_LANGUAGE"] = "en"

    target = '%s (while bundling under "%s")' % (bundle_name_with_extension, str(rule_label))

    control = struct(
        binary = True,
        output = output_plist.path,
        plists = [p.path for p in input_files],
        target = target,
        variable_substitutions = struct(**substitutions),
    )

    control_file = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = output_discriminator,
        file_name = paths.join(bundle_name_with_extension, "%s-control" % output_plist.basename),
    )
    actions.write(
        output = control_file,
        content = json.encode(control),
    )

    plisttool_action(
        actions = actions,
        control_file = control_file,
        inputs = input_files,
        mnemonic = "CompileInfoPlist",
        outputs = [output_plist],
        platform_prerequisites = platform_prerequisites,
        plisttool = plisttool,
    )

def merge_root_infoplists(
        *,
        actions,
        bundle_name,
        bundle_id = None,
        bundle_extension,
        executable_name = None,
        child_plists = [],
        child_required_values = [],
        environment_plist,
        extensionkit_keys_required = False,
        include_executable_name = True,
        input_plists,
        launch_storyboard,
        output_discriminator,
        output_plist,
        output_pkginfo,
        platform_prerequisites,
        plisttool,
        rule_descriptor,
        rule_label,
        version,
        version_keys_required = False):
    """Creates an action that merges Info.plists and converts them to binary.

    This action merges multiple plists by shelling out to plisttool, then
    compiles the final result into a single binary plist file.

    Args:
      actions: The actions provider from `ctx.actions`.
      bundle_name: The name of the output bundle.
      bundle_id: The bundle identifier to set in the output plist.
      bundle_extension: The extension for the bundle.
      executable_name: The name of the output executable.
      child_plists: A list of plists from child targets (such as extensions
          or Watch apps) whose bundle IDs and version strings should be
          validated against the compiled plist for consistency.
      child_required_values: A list of pairs containing a client target plist
          and the pairs to check. For more information on the second item in the
          pair, see plisttool's `child_plist_required_values`, as this is passed
          straight through to it.
      environment_plist: An executable file referencing the environment_plist tool.
      extensionkit_keys_required: If True, the merged Info.plist file must include entries for
          EXAppExtensionAttributes and EXExtensionPointIdentifier, and have no NSExtension key.
      include_executable_name: If True, the executable name will be added to
          the plist in the `CFBundleExecutable` key. This is mainly intended for
          plists embedded in a command line tool which don't need this value.
      input_plists: The root plist files to merge.
      launch_storyboard: A file to be used as a launch screen for the application.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      output_pkginfo: The file reference for the PkgInfo file. Can be None if not
          required.
      output_plist: The file reference for the merged output plist.
      platform_prerequisites: Struct containing information on the platform being targeted.
      plisttool: A files_to_run for the plist tool.
      rule_descriptor: A rule descriptor for platform and product types from the rule context.
      rule_label: The label of the target being analyzed.
      version: A label referencing AppleBundleVersionInfo, if provided by the rule.
      version_keys_required: If True, the merged Info.plist file must include
          entries for CFBundleShortVersionString and CFBundleVersion.
    """
    input_files = list(input_plists + child_plists)

    # plists and forced_plists are lists of plist representations that should be
    # merged into the final Info.plist. Values in plists will be validated to be
    # unique, while values in forced_plists are forced into the final Info.plist,
    # without validation. Each array can contain either a path to a plist file to
    # merge, or a struct that represents the values of the plist to merge.
    plists = [p.path for p in input_plists]
    forced_plists = []

    # plisttool options for merging the Info.plist file.
    info_plist_options = {}

    bundle_name_with_extension = bundle_name + bundle_extension
    product_name = paths.replace_extension(bundle_name_with_extension, "")

    # Values for string replacement substitutions to perform in the merged
    # Info.plist
    substitutions = {
        "BUNDLE_NAME": bundle_name_with_extension,
        "PRODUCT_NAME": product_name,
    }

    # The default in Xcode is for PRODUCT_NAME and TARGET_NAME to be the same.
    # Support TARGET_NAME for substitutions even though it might not be the
    # target name in the BUILD file.
    substitutions["TARGET_NAME"] = product_name

    # The generated Info.plists from Xcode's project templates use
    # DEVELOPMENT_LANGUAGE as the default variable substitution for
    # CFBundleDevelopmentRegion. We substitute this to `en` to support
    # Info.plists out of the box coming from Xcode.
    substitutions["DEVELOPMENT_LANGUAGE"] = "en"

    executable_name = executable_name or bundle_name
    if include_executable_name:
        substitutions["EXECUTABLE_NAME"] = executable_name
        forced_plists.append(struct(CFBundleExecutable = executable_name))

    if bundle_id:
        substitutions["PRODUCT_BUNDLE_IDENTIFIER"] = bundle_id

        # Pass the bundle_id as a plist and not a force_plist, this way the
        # merging will validate that any existing value matches. Historically
        # mismatches between the input Info.plist and rules bundle_id have
        # been valid bugs, so this will still catch that.
        plists.append(struct(CFBundleIdentifier = bundle_id))

    if child_plists:
        info_plist_options["child_plists"] = struct(
            **{str(p.owner): p.path for p in child_plists}
        )

    if child_required_values:
        info_plist_options["child_plist_required_values"] = struct(
            **{str(p.owner): v for (p, v) in child_required_values}
        )

    if extensionkit_keys_required:
        info_plist_options["extensionkit_keys_required"] = True

    if (version != None and AppleBundleVersionInfo in version):
        version_info = version[AppleBundleVersionInfo]
        input_files.append(version_info.version_file)
        info_plist_options["version_file"] = version_info.version_file.path

    if version_keys_required:
        info_plist_options["version_keys_required"] = True

    # Keys to be forced into the Info.plist file.
    # b/67853874 - move this to the right platform specific rule(s).
    if launch_storyboard:
        short_name = paths.split_extension(launch_storyboard.basename)[0]
        forced_plists.append(struct(UILaunchStoryboardName = short_name))

    # Add any UIDeviceFamily entry needed.
    families = platform_support.ui_device_family_plist_value(
        platform_prerequisites = platform_prerequisites,
    )
    if families:
        forced_plists.append(struct(UIDeviceFamily = families))

    # Collect any values for special product types that we have to manually put
    # in (duplicating what Xcode apparently does under the hood).
    if rule_descriptor.additional_infoplist_values:
        forced_plists.append(
            struct(**rule_descriptor.additional_infoplist_values),
        )

    # Replace PRODUCT_BUNDLE_PACKAGE_TYPE based on info in rule descriptor
    if rule_descriptor.bundle_package_type:
        substitutions["PRODUCT_BUNDLE_PACKAGE_TYPE"] = rule_descriptor.bundle_package_type

    if platform_prerequisites.platform_type == apple_common.platform_type.macos:
        plist_key = "LSMinimumSystemVersion"
    else:
        plist_key = "MinimumOSVersion"

    if environment_plist:
        input_files.append(environment_plist)
        forced_plists.append(environment_plist.path)

    platform = platform_prerequisites.platform
    sdk_version = platform_prerequisites.sdk_version
    platform_with_version = platform.name_in_plist.lower() + str(sdk_version)
    forced_plists.append(
        struct(
            CFBundleSupportedPlatforms = [platform.name_in_plist],
            DTPlatformName = platform.name_in_plist.lower(),
            DTSDKName = platform_with_version,
            **{plist_key: platform_prerequisites.minimum_deployment_os}
        ),
    )

    output_files = [output_plist]
    if output_pkginfo:
        info_plist_options["pkginfo"] = output_pkginfo.path
        output_files.append(output_pkginfo)

    control = struct(
        binary = rule_descriptor.binary_infoplist,
        forced_plists = forced_plists,
        info_plist_options = struct(**info_plist_options),
        output = output_plist.path,
        plists = plists,
        target = str(rule_label),
        variable_substitutions = struct(**substitutions),
    )

    control_file = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = output_discriminator,
        file_name = "%s-root-control" % output_plist.basename,
    )
    actions.write(
        output = control_file,
        content = json.encode(control),
    )

    plisttool_action(
        actions = actions,
        control_file = control_file,
        inputs = input_files,
        mnemonic = "CompileRootInfoPlist",
        outputs = output_files,
        platform_prerequisites = platform_prerequisites,
        plisttool = plisttool,
    )
