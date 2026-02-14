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

"""Actions related to codesigning."""

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
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:rule_support.bzl",
    "rule_support",
)
load(
    "//apple/internal/utils:defines.bzl",
    "defines",
)

# The adhoc signature used as an identity, partially documented at https://developer.apple.com/documentation/security/seccodesignatureflags/1397793-adhoc
_ADHOC_PSEUDO_IDENTITY = "-"

def _double_quote(raw_string):
    """Add double quotes around the string and preserve existing quote characters.

    Args:
      raw_string: A string that might have shell-syntaxed environment variables.

    Returns:
      The string with double quotes.
    """
    return "\"" + raw_string.replace("\"", "\\\"") + "\""

def _no_op(x):
    """Helper that does not nothing be return the result."""
    return x

def _codesignopts_from_rule_ctx(ctx):
    """Get the expanded codesign opts from the rule context

    Args:
      ctx: The rule context to fetch the options and expand from

    Returns:
      The list of codesignopts
    """
    return [
        ctx.expand_make_variables("codesignopts", opt, {})
        for opt in ctx.attr.codesignopts
    ]

def _preferred_codesigning_identity(platform_prerequisites):
    """Returns the preferred codesigning identity from platform prerequisites"""
    if not platform_prerequisites.platform.is_device:
        return _ADHOC_PSEUDO_IDENTITY
    build_settings = platform_prerequisites.build_settings
    if build_settings:
        objc_fragment = platform_prerequisites.objc_fragment
        if objc_fragment:
            # TODO(b/252873771): Remove this fallback when the native Bazel flag
            # ios_signing_cert_name is removed.
            return (build_settings.signing_certificate_name or
                    objc_fragment.signing_certificate_name)
        else:
            return build_settings.signing_certificate_name
    return None

def _codesign_args_for_path(
        *,
        codesignopts,
        entitlements_file,
        path_to_sign,
        platform_prerequisites,
        provisioning_profile,
        shell_quote = True):
    """Returns a command line for the codesigning tool wrapper script.

    Args:
      codesignopts: Extra options to pass to the `codesign` tool
      entitlements_file: The entitlements file to pass to codesign. May be `None`
          for non-app binaries (e.g. test bundles).
      path_to_sign: A struct indicating the path that should be signed and its
          optionality (see `_path_to_sign`).
      platform_prerequisites: Struct containing information on the platform being targeted.
      provisioning_profile: The provisioning profile file. May be `None`.
      shell_quote: Sanitizes the arguments to be evaluated in a shell.

    Returns:
      The codesign command invocation for the given directory as a list.
    """
    if not path_to_sign.is_directory and path_to_sign.signed_frameworks:
        fail("Internal Error: Received a list of signed frameworks as exceptions " +
             "for code signing, but path to sign is not a directory.")

    for x in path_to_sign.signed_frameworks:
        if not x.startswith(path_to_sign.path):
            fail("Internal Error: Signed framework does not have the current path " +
                 "to sign (%s) as its prefix (%s)." % (path_to_sign.path, x))

    cmd_codesigning = [
        "--codesign",
        "/usr/bin/codesign",
    ]

    # Add quotes for sanitizing inputs when they're invoked directly from a shell script, for
    # instance when using this string to assemble the output of codesigning_command.
    maybe_quote = shell.quote if shell_quote else _no_op
    maybe_double_quote = _double_quote if shell_quote else _no_op

    # First, try to use the identity passed on the command line, if any. If it's a simulator build,
    # use an ad hoc identity.
    identity = _preferred_codesigning_identity(platform_prerequisites)
    if not identity:
        if provisioning_profile:
            cmd_codesigning.extend([
                "--mobileprovision",
                maybe_quote(provisioning_profile.path),
            ])

        else:
            identity = _ADHOC_PSEUDO_IDENTITY

    if identity:
        cmd_codesigning.extend([
            "--identity",
            maybe_quote(identity),
        ])

    # The entitlements rule ensures that entitlements_file is None or a file
    # containing only "com.apple.security.get-task-allow" when building for the
    # simulator.
    if path_to_sign.use_entitlements and entitlements_file:
        cmd_codesigning.extend([
            "--entitlements",
            maybe_quote(entitlements_file.path),
        ])

    if platform_prerequisites.platform.is_device:
        cmd_codesigning.append("--force")
    else:
        cmd_codesigning.extend([
            "--force",
            "--disable_timestamp",
        ])

    if path_to_sign.is_directory:
        cmd_codesigning.append("--directory_to_sign")
    else:
        cmd_codesigning.append("--target_to_sign")

    # Because the path does include environment variables which need to be expanded, path has to be
    # quoted using double quotes, this means that path can't be quoted using shell.quote.
    cmd_codesigning.append(maybe_double_quote(path_to_sign.path))

    if path_to_sign.signed_frameworks:
        for signed_framework in path_to_sign.signed_frameworks:
            # Signed frameworks must also be double quoted, as they too have an environment
            # variable to be expanded.
            cmd_codesigning.extend([
                "--signed_path",
                maybe_double_quote(signed_framework),
            ])

    cmd_codesigning.append("--")
    cmd_codesigning.extend(codesignopts)
    return cmd_codesigning

def _path_to_sign(*, path, is_directory = False, signed_frameworks = [], use_entitlements = True):
    """Returns a "path to sign" value to be passed to `_signing_command_lines`.

    Args:
      path: The path to sign, relative to wherever the code signing command lines
          are being executed.
      is_directory: If `True`, the path is a directory and not a bundle, indicating
          that the contents of each item in the directory should be code signed
          except for the invisible files prefixed with a period.
      signed_frameworks: If provided, a list of frameworks that have already been signed.
      use_entitlements: If provided, indicates if the entitlements on the bundling
          target should be used for signing this path (useful to disabled the use
          when signing frameworks within an iOS app).

    Returns:
      A `struct` that can be passed to `_signing_command_lines`.
    """
    return struct(
        path = path,
        is_directory = is_directory,
        signed_frameworks = signed_frameworks,
        use_entitlements = use_entitlements,
    )

def _validate_provisioning_profile(
        *,
        rule_descriptor,
        platform_prerequisites,
        provisioning_profile):
    # Verify that a provisioning profile was provided for device builds on
    # platforms that require it.
    if (platform_prerequisites.platform.is_device and
        rule_descriptor.requires_signing_for_device and
        not provisioning_profile):
        fail("The provisioning_profile attribute must be set for device " +
             "builds on this platform (%s)." %
             platform_prerequisites.platform_type)

def _signing_command_lines(
        *,
        codesigningtool,
        codesignopts,
        entitlements_file,
        paths_to_sign,
        platform_prerequisites,
        provisioning_profile):
    """Returns a multi-line string with codesign invocations for the bundle.

    For any signing identity other than ad hoc, the identity is verified as being
    valid in the keychain and an error will be emitted if the identity cannot be
    used for signing for any reason.

    Args:
      codesigningtool: The executable `File` representing the code signing tool.
      codesignopts: Extra options to pass to the `codesign` tool
      entitlements_file: The entitlements file to pass to codesign.
      paths_to_sign: A list of values returned from `path_to_sign` that indicate
          paths that should be code-signed.
      platform_prerequisites: Struct containing information on the platform being targeted.
      provisioning_profile: The provisioning profile file. May be `None`.

    Returns:
      A multi-line string with codesign invocations for the bundle.
    """
    commands = []

    # Use of the entitlements file is not recommended for the signing of frameworks. As long as
    # this remains the case, we do have to split the "paths to sign" between multiple invocations
    # of codesign.
    for path_to_sign in paths_to_sign:
        codesign_command = [codesigningtool.path]
        codesign_command.extend(_codesign_args_for_path(
            entitlements_file = entitlements_file,
            path_to_sign = path_to_sign,
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            codesignopts = codesignopts,
        ))
        commands.append(" ".join(codesign_command))
    return "\n".join(commands)

def _should_sign_simulator_frameworks(
        *,
        features):
    """Check if simulator bound framework bundles should be codesigned.

    Args:

    Returns:
      True/False for if the framework should be signed.
    """
    if "apple.skip_codesign_simulator_bundles" in features:
        return False

    # To preserve existing functionality, where Frameworks/* bundles are always
    # signed, we only skip them with the new flag. This check will go away when
    # `apple.codesign_simulator_bundles` goes away.
    return True

def _should_sign_simulator_bundles(
        *,
        config_vars,
        features,
        rule_descriptor):
    """Check if a main bundle should be codesigned.

    Args:

    Returns:
      True/False for if the bundle should be signed.

    """
    if "apple.codesign_simulator_bundles" in config_vars:
        # buildifier: disable=print
        print("warning: --define apple.codesign_simulator_bundles is deprecated, please switch to --features=apple.skip_codesign_simulator_bundles")

    if "apple.skip_codesign_simulator_bundles" in features:
        return False

    if not rule_descriptor.skip_simulator_signing_allowed:
        return True

    # Default is to sign.
    return defines.bool_value(
        config_vars = config_vars,
        define_name = "apple.codesign_simulator_bundles",
        default = True,
    )

def _should_sign_bundles(*, provisioning_profile, rule_descriptor, features):
    should_sign_bundles = True

    codesigning_exceptions = rule_descriptor.codesigning_exceptions
    if "disable_legacy_signing" in features:
        should_sign_bundles = False
    elif (codesigning_exceptions ==
          rule_support.codesigning_exceptions.sign_with_provisioning_profile):
        # If the rule doesn't have a provisioning profile, do not sign the binary or its
        # frameworks.
        if (not provisioning_profile and
            "apple.codesign_frameworks_without_provisioning_profile" not in features):
            should_sign_bundles = False
    elif codesigning_exceptions == rule_support.codesigning_exceptions.skip_signing:
        should_sign_bundles = False
    elif codesigning_exceptions != rule_support.codesigning_exceptions.none:
        fail("Internal Error: Encountered unsupported state for codesigning_exceptions.")

    return should_sign_bundles

def _codesigning_args(
        *,
        entitlements,
        features,
        full_archive_path,
        is_framework = False,
        platform_prerequisites,
        provisioning_profile,
        rule_descriptor):
    """Returns a set of codesigning arguments to be passed to the codesigning tool.

    Args:
        entitlements: The entitlements file to sign with. Can be None.
        features: List of features enabled by the user. Typically from `ctx.features`.
        full_archive_path: The full path to the codesigning target.
        is_framework: If the target is a framework. False by default.
        platform_prerequisites: Struct containing information on the platform being targeted.
        provisioning_profile: File for the provisioning profile.
        rule_descriptor: A rule descriptor for platform and product types from the rule context.

    Returns:
        A list containing the arguments to pass to the codesigning tool.
    """
    should_sign_bundles = _should_sign_bundles(
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        features = features,
    )
    if not should_sign_bundles:
        return []

    is_device = platform_prerequisites.platform.is_device
    should_sign_sim_bundles = _should_sign_simulator_bundles(
        config_vars = platform_prerequisites.config_vars,
        features = platform_prerequisites.features,
        rule_descriptor = rule_descriptor,
    )

    # We need to re-sign imported frameworks
    if not is_framework and not is_device and not should_sign_sim_bundles:
        return []

    _validate_provisioning_profile(
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
    )

    return _codesign_args_for_path(
        entitlements_file = entitlements,
        path_to_sign = _path_to_sign(path = full_archive_path),
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        shell_quote = False,
        codesignopts = [],
    )

def _codesigning_command(
        *,
        bundle_path = "",
        codesigningtool,
        codesignopts,
        entitlements,
        features,
        frameworks_path,
        platform_prerequisites,
        provisioning_profile,
        rule_descriptor,
        signed_frameworks):
    """Returns a codesigning command that includes framework embedded bundles.

    Args:
        bundle_path: The location of the bundle, relative to the archive.
        codesigningtool: The executable `File` representing the code signing tool.
        codesignopts: Extra options to pass to the `codesign` tool
        entitlements: The entitlements file to sign with. Can be None.
        features: List of features enabled by the user. Typically from `ctx.features`.
        frameworks_path: The location of the Frameworks directory, relative to the archive.
        platform_prerequisites: Struct containing information on the platform being targeted.
        provisioning_profile: File for the provisioning profile.
        rule_descriptor: A rule descriptor for platform and product types from the rule context.
        signed_frameworks: A depset containing each framework that has already been signed.

    Returns:
        A string containing the codesigning commands.
    """
    should_sign_bundles = _should_sign_bundles(
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        features = features,
    )
    if not should_sign_bundles:
        return ""

    _validate_provisioning_profile(
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
    )
    paths_to_sign = []
    is_device = platform_prerequisites.platform.is_device

    # The command returned by this function is executed as part of a bundling shell script.
    # Each directory to be signed must be prefixed by $WORK_DIR, which is the variable in that
    # script that contains the path to the directory where the bundle is being built.
    should_sign_sim_frameworks = _should_sign_simulator_frameworks(
        features = platform_prerequisites.features,
    )
    if (frameworks_path and should_sign_sim_frameworks) or is_device:
        framework_root = paths.join("$WORK_DIR", frameworks_path) + "/"
        full_signed_frameworks = []

        for signed_framework in signed_frameworks.to_list():
            full_signed_frameworks.append(paths.join(framework_root, signed_framework))

        paths_to_sign.append(
            _path_to_sign(
                path = framework_root,
                is_directory = True,
                signed_frameworks = full_signed_frameworks,
                use_entitlements = False,
            ),
        )
    should_sign_sim_bundles = _should_sign_simulator_bundles(
        config_vars = platform_prerequisites.config_vars,
        features = platform_prerequisites.features,
        rule_descriptor = rule_descriptor,
    )
    if is_device or should_sign_sim_bundles:
        path_to_sign = paths.join("$WORK_DIR", bundle_path)
        paths_to_sign.append(
            _path_to_sign(path = path_to_sign),
        )
    return _signing_command_lines(
        codesigningtool = codesigningtool,
        entitlements_file = entitlements,
        paths_to_sign = paths_to_sign,
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        codesignopts = codesignopts,
    )

def _generate_codesigning_dossier_action(
        actions,
        label_name,
        dossier_codesigningtool,
        embedded_dossiers,
        entitlements,
        output_discriminator,
        output_dossier,
        platform_prerequisites,
        provisioning_profile):
    """Generates a codesigning dossier based on parameters.

    Args:
      actions: The actions provider from `ctx.actions`.
      label_name: Name of the target being built.
      dossier_codesigningtool: The files_to_run for the code signing tool.
      embedded_dossiers: An optional List of Structs generated from
         `embedded_codesigning_dossier` that should also be included in this
          dossier.
      entitlements: Optional file representing the entitlements to sign with.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      output_dossier: The `File` representing the output dossier file - the zipped dossier will be placed here.
      platform_prerequisites: Struct containing information on the platform being targeted.
      provisioning_profile: The provisioning profile file. May be `None`.
    """
    input_files = [x.dossier_file for x in embedded_dossiers]

    mnemonic = "GenerateCodesigningDossier"
    progress_message = "Generating codesigning dossier for %s" % label_name

    dossier_arguments = ["--output", output_dossier.path, "--zip"]

    # Try to use the identity passed on the command line, if any. If it's a simulator build, use an
    # ad hoc identity.
    codesign_identity = _preferred_codesigning_identity(platform_prerequisites)
    if not codesign_identity and not provisioning_profile:
        codesign_identity = _ADHOC_PSEUDO_IDENTITY
    if codesign_identity:
        dossier_arguments.extend(["--codesign_identity", codesign_identity])
    else:
        dossier_arguments.append("--infer_identity")
    if entitlements and platform_prerequisites.platform.is_device:
        # Entitlements are embedded as segments of the linked simulator binary. They should not be
        # used for signing simulator binaries.
        input_files.append(entitlements)
        dossier_arguments.extend(["--entitlements_file", entitlements.path])
    if provisioning_profile and codesign_identity != _ADHOC_PSEUDO_IDENTITY:
        # If we're signing with the ad-hoc pseudo-identity, no identity may be retrieved from the
        # signed artifact and any code requirement placing restrictions on the signing identity will
        # fail.
        #
        # Therefore, restrictions placed from the provisioning profile will effectively break an
        # ad-hoc signed artifact, and will always result in failing code signing checks on the
        # bundle or binary.
        #
        # Only reference and embed the provisioning profile in standard code signing.
        input_files.append(provisioning_profile)
        dossier_arguments.extend(["--provisioning_profile", provisioning_profile.path])

    for embedded_dossier in embedded_dossiers:
        input_files.append(embedded_dossier.dossier_file)
        dossier_arguments.extend(["--embedded_dossier", embedded_dossier.relative_bundle_path, embedded_dossier.dossier_file.path])

    args_file = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "dossier_arguments",
    )
    actions.write(
        output = args_file,
        content = "\n".join(dossier_arguments),
    )

    input_files.append(args_file)
    args_path_argument = "@%s" % args_file.path
    args = ["create", args_path_argument]

    apple_support.run(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        arguments = args,
        executable = dossier_codesigningtool,
        inputs = input_files,
        mnemonic = mnemonic,
        outputs = [output_dossier],
        progress_message = progress_message,
        xcode_config = platform_prerequisites.xcode_version_config,
    )

def _post_process_and_sign_archive_action(
        *,
        actions,
        archive_codesigning_path,
        codesign_inputs,
        codesigningtool,
        codesignopts,
        entitlements = None,
        features,
        frameworks_path,
        input_archive,
        ipa_post_processor,
        label_name,
        output_archive,
        output_archive_root_path,
        output_discriminator,
        platform_prerequisites,
        process_and_sign_template,
        provisioning_profile,
        rule_descriptor,
        signed_frameworks):
    """Post-processes and signs an archived bundle.

    Args:
      actions: The actions provider from `ctx.actions`.
      archive_codesigning_path: The codesigning path relative to the archive.
      codesign_inputs: Extra inputs needed for the `codesign` tool.
      codesigningtool: The files_to_run for the code signing tool.
      codesignopts: Extra options to pass to the `codesign` tool.
      entitlements: Optional file representing the entitlements to sign with.
      features: List of features enabled by the user. Typically from `ctx.features`.
      frameworks_path: The Frameworks path relative to the archive.
      input_archive: The `File` representing the archive containing the bundle
          that has not yet been processed or signed.
      ipa_post_processor: A file that acts as a bundle post processing tool. May be `None`.
      label_name: Name of the target being built.
      output_archive: The `File` representing the processed and signed archive.
      output_archive_root_path: The `string` path to where the processed, uncompressed archive
          should be located.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      platform_prerequisites: Struct containing information on the platform being targeted.
      process_and_sign_template: A template for a shell script to process and sign as a file.
      provisioning_profile: The provisioning profile file. May be `None`.
      rule_descriptor: A rule descriptor for platform and product types from the rule context.
      signed_frameworks: Depset containing each framework that has already been signed.
    """
    input_files = [input_archive]
    processing_tools = []

    execution_requirements = {
        # Unsure, but may be needed for keychain access, especially for files
        # that live in $HOME.
        "no-sandbox": "1",
    }

    signing_command_lines = _codesigning_command(
        bundle_path = archive_codesigning_path,
        codesigningtool = codesigningtool.executable,
        codesignopts = codesignopts,
        entitlements = entitlements,
        features = features,
        frameworks_path = frameworks_path,
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        signed_frameworks = signed_frameworks,
    )
    if signing_command_lines:
        processing_tools.append(codesigningtool)
        if entitlements:
            input_files.append(entitlements)
        if provisioning_profile:
            input_files.append(provisioning_profile)
            if platform_prerequisites.platform.is_device:
                # Added so that the output of this action is not cached
                # remotely, in case multiple developers sign the same artifact
                # with different identities.
                execution_requirements["no-remote"] = "1"

    ipa_post_processor_path = ""
    if ipa_post_processor:
        processing_tools.append(ipa_post_processor)
        ipa_post_processor_path = ipa_post_processor.path

    # Compress the IPA when requested. By default, enable compression for optimized (release) builds
    # to reduce file size, and disable compression for debug builds to speed up the build.
    config_vars = platform_prerequisites.config_vars
    should_compress = defines.bool_value(
        config_vars = config_vars,
        define_name = "apple.compress_ipa",
        default = (config_vars["COMPILATION_MODE"] == "opt"),
    )

    # TODO(b/163217926): These are kept the same for the three different actions
    # that could be run to ensure anything keying off these values continues to
    # work. After some data is collected, the values likely can be revisited and
    # changed.
    mnemonic = "ProcessAndSign"
    progress_message = "Processing and signing %s" % label_name

    # If there is no work to be done, skip the processing/signing action, just
    # copy the file over.
    has_work = any([signing_command_lines, ipa_post_processor_path, should_compress])
    if not has_work:
        actions.run_shell(
            command = "cp -p '%s' '%s'" % (input_archive.path, output_archive.path),
            inputs = [input_archive],
            mnemonic = mnemonic,
            outputs = [output_archive],
            progress_message = progress_message,
        )
        return

    process_and_sign_expanded_template = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "process-and-sign-%s.sh" % hash(output_archive.path),
    )
    actions.expand_template(
        template = process_and_sign_template,
        output = process_and_sign_expanded_template,
        is_executable = True,
        substitutions = {
            "%ipa_post_processor%": shell.quote(ipa_post_processor_path) or "",
            "%output_path%": shell.quote(output_archive.path),
            "%should_compress%": "1" if should_compress else "",
            "%should_verify%": "1",  # always verify the crc
            "%signing_command_lines%": signing_command_lines,
            "%unprocessed_archive_path%": shell.quote(input_archive.path),
            "%work_dir%": shell.quote(output_archive_root_path),
        },
    )

    # Build up some arguments for the script to allow logging to tell what work
    # is being done within the action's script.
    arguments = []
    if signing_command_lines:
        arguments.append("should_sign")
    if ipa_post_processor_path:
        arguments.append("should_process")
    if should_compress:
        arguments.append("should_compress")

    run_on_darwin = any([signing_command_lines, ipa_post_processor_path])
    if run_on_darwin:
        apple_support.run(
            actions = actions,
            apple_fragment = platform_prerequisites.apple_fragment,
            arguments = arguments,
            executable = process_and_sign_expanded_template,
            execution_requirements = execution_requirements,
            inputs = input_files + codesign_inputs,
            mnemonic = mnemonic,
            outputs = [output_archive],
            progress_message = progress_message,
            tools = processing_tools,
            xcode_config = platform_prerequisites.xcode_version_config,
        )
    else:
        actions.run(
            arguments = arguments,
            executable = process_and_sign_expanded_template,
            inputs = input_files,
            mnemonic = mnemonic,
            outputs = [output_archive],
            progress_message = progress_message,
        )

def _sign_binary_action(
        *,
        actions,
        codesign_inputs,
        codesigningtool,
        codesignopts,
        input_binary,
        output_binary,
        platform_prerequisites,
        provisioning_profile,
        rule_descriptor):
    """Signs the input binary file, copying it into the given output binary file.

    Args:
      actions: The actions provider from `ctx.actions`.
      codesign_inputs: Extra inputs needed for the `codesign` tool.
      codesigningtool: The files_to_run for the code signing tool.
      codesignopts: Extra options to pass to the `codesign` tool.
      input_binary: The `File` representing the binary to be signed.
      output_binary: The `File` representing signed binary.
      platform_prerequisites: Struct containing information on the platform being targeted.
      provisioning_profile: The provisioning profile file. May be `None`.
      rule_descriptor: A rule descriptor for platform and product types from the rule context.
    """
    _validate_provisioning_profile(
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
    )

    # It's not hermetic to sign the binary that was built by the apple_binary
    # target that this rule takes as an input, so we copy it and then execute the
    # code signing commands on that copy in the same action.
    path_to_sign = _path_to_sign(path = output_binary.path)
    signing_commands = _signing_command_lines(
        codesigningtool = codesigningtool.executable,
        entitlements_file = None,
        paths_to_sign = [path_to_sign],
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        codesignopts = codesignopts,
    )

    execution_requirements = {
        # Unsure, but may be needed for keychain access, especially for files
        # that live in $HOME.
        "no-sandbox": "1",
    }
    if platform_prerequisites.platform.is_device and provisioning_profile:
        # Added so that the output of this action is not cached remotely,
        # in case multiple developers sign the same artifact with different
        # identities.
        execution_requirements["no-remote"] = "1"

    apple_support.run_shell(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        command = "cp {input_binary} {output_binary}".format(
            input_binary = input_binary.path,
            output_binary = output_binary.path,
        ) + "\n" + signing_commands,
        execution_requirements = execution_requirements,
        inputs = [input_binary] + codesign_inputs,
        mnemonic = "SignBinary",
        outputs = [output_binary],
        tools = [codesigningtool],
        xcode_config = platform_prerequisites.xcode_version_config,
    )

def _embedded_codesigning_dossier(relative_bundle_path, dossier_file):
    """Returns a struct describing a dossier to be embedded in another dossier.

    Args:
      dossier_file: The File representing the zipped dossier to be embedded.
      relative_bundle_path: The string path of the artifact this dossier
        describes relative to the root of the bundle it is embedded in.
        E.g. 'PlugIns/NetworkExtension.appex'
    """
    return struct(relative_bundle_path = relative_bundle_path, dossier_file = dossier_file)

codesigning_support = struct(
    codesigning_args = _codesigning_args,
    codesigning_command = _codesigning_command,
    codesignopts_from_rule_ctx = _codesignopts_from_rule_ctx,
    embedded_codesigning_dossier = _embedded_codesigning_dossier,
    generate_codesigning_dossier_action = _generate_codesigning_dossier_action,
    post_process_and_sign_archive_action = _post_process_and_sign_archive_action,
    should_sign_bundles = _should_sign_bundles,
    sign_binary_action = _sign_binary_action,
)
