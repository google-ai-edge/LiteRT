# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""Implementation of the xcframework rules."""

load("@bazel_skylib//lib:partial.bzl", "partial")
load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load("@build_bazel_rules_swift//swift:swift.bzl", "SwiftInfo")
load(
    "//apple:providers.bzl",
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
    "//apple/internal:cc_info_support.bzl",
    "cc_info_support",
)
load(
    "//apple/internal:experimental.bzl",
    "is_experimental_tree_artifact_enabled",
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
    "//apple/internal:outputs.bzl",
    "outputs",
)
load(
    "//apple/internal:partials.bzl",
    "partials",
)
load(
    "//apple/internal:platform_support.bzl",
    "platform_support",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebundleinfo",
    "new_applestaticxcframeworkbundleinfo",
    "new_applexcframeworkbundleinfo",
)
load(
    "//apple/internal:resources.bzl",
    "resources",
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
    "//apple/internal:rule_support.bzl",
    "rule_support",
)
load(
    "//apple/internal:swift_support.bzl",
    "swift_support",
)
load(
    "//apple/internal:transition_support.bzl",
    "transition_support",
)
load(
    "//apple/internal/aspects:resource_aspect.bzl",
    "apple_resource_aspect",
)
load(
    "//apple/internal/aspects:swift_usage_aspect.bzl",
    "swift_usage_aspect",
)
load(
    "//apple/internal/utils:files.bzl",
    "files",
)

def _has_non_system_swift_modules(*, target):
    """Indicates if the given target references any non-system Swift modules.

    This is a reasonable signal to determine if we need to generate framework interfaces, though
    correctness should be determined as well via further analysis of the graph of deps. See
    b/321089167 for follow up work to that end.

    Args:
        target: A Target representing a dep for a given split from `deps` on the XCFramework rule.

    Returns:
        `True` if a non-system module was found from the target's SwiftInfo provider, `False`
        otherwise.
    """
    if SwiftInfo not in target:
        return False

    swift_info = target[SwiftInfo]

    # Covers both direct and transitive modules, from how the SwiftInfo provider is constructed.
    for module in swift_info.transitive_modules.to_list():
        if module.swift and not module.is_system:
            return True

    return False

def _group_link_outputs_by_library_identifier(
        *,
        actions,
        apple_fragment,
        deps,
        label_name,
        link_result,
        xcode_config):
    """Groups linking outputs by library identifier with additional platform information.

    Linking outputs artifacts are combined using the lipo tool if necessary due to grouping.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        deps: Label list of dependencies from rule context (ctx.split_attr.deps).
        label_name: Name of the target being built.
        link_result: The struct returned by `linking_support.register_binary_linking_action`.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.

    Returns:
        A list of structs with the following fields; `architectures` containing a list of the
        architectures that the binary was built with, `binary` referencing the output binary linked
        with the `lipo` tool if necessary, or referencing a symlink to the original binary if not,
        `dsym_binaries` which is a mapping of architectures to dsym binaries if any were created,
        `environment` to reference the target environment the binary was built for, `linkmaps` which
        is a mapping of architectures to linkmaps if any were created, and `platform` to reference
        the target platform the binary was built for.
    """
    linking_type = None
    for attr_name in ["binary", "library"]:
        if hasattr(link_result, attr_name):
            linking_type = attr_name
            break

    if not linking_type:
        fail("Apple linking APIs output struct must define either 'binary' or 'library'.\n" +
             "This is most likely a rules_apple bug, please file a bug with reproduction steps.")

    # Organize each output as a platform_environment, where each can accept one or more archs.
    link_outputs_by_framework = {}

    # Iterate through the outputs of the registered linking action, match archs to platform and
    # environment combinations.
    for link_output in link_result.outputs:
        framework_key = link_output.platform + "_" + link_output.environment
        if link_outputs_by_framework.get(framework_key):
            link_outputs_by_framework[framework_key].append(link_output)
        else:
            link_outputs_by_framework[framework_key] = [link_output]

    link_outputs_by_library_identifier = {}

    # Iterate through the structure again, this time creating a structure equivalent to link_result
    # .outputs but with .architecture replaced with .architectures, .dsym_binary replaced with
    # .dsym_binaries, and .linkmap replaced with .linkmaps
    for framework_key, link_outputs in link_outputs_by_framework.items():
        inputs = [getattr(output, linking_type) for output in link_outputs]
        filename = "{}_{}".format(label_name, framework_key)
        extension = inputs[0].extension
        if extension != "":
            filename = "{}.{}".format(filename, extension)
        fat_binary = actions.declare_file(filename)
        linking_support.lipo_or_symlink_inputs(
            actions = actions,
            inputs = inputs,
            output = fat_binary,
            apple_fragment = apple_fragment,
            xcode_config = xcode_config,
        )

        architectures = []
        dsym_binaries = {}
        linkmaps = {}
        split_attr_keys = []
        framework_swift_infos = {}
        uses_swift = False
        for link_output in link_outputs:
            split_attr_key = transition_support.xcframework_split_attr_key(
                arch = link_output.architecture,
                environment = link_output.environment,
                platform_type = link_output.platform,
            )

            architectures.append(link_output.architecture)
            split_attr_keys.append(split_attr_key)

            # Determine up front if the given dep references any SwiftUsageInfo, for partial
            # processing.
            if swift_support.uses_swift(deps[split_attr_key]):
                uses_swift = True

            # If there's any Swift dependencies on this framework rule, look for providers
            # referencing non-system Swift modules to see if we need to generate Swift interfaces.
            for dep in deps[split_attr_key]:
                # TODO(b/321089167): Fail the build if the build graph has an arrangement of Swift
                # modules that is not suitable for generating frameworks.
                if _has_non_system_swift_modules(target = dep):
                    framework_swift_infos[link_output.architecture] = dep[SwiftInfo]

            # static library linking does not support dsym, and linkmaps yet.
            if linking_type == "binary":
                dsym_binaries[link_output.architecture] = link_output.dsym_binary
                linkmaps[link_output.architecture] = link_output.linkmap

        environment = link_outputs[0].environment
        platform = link_outputs[0].platform

        library_identifier = _library_identifier(
            architectures = architectures,
            environment = environment,
            platform = platform,
        )

        link_outputs_by_library_identifier[library_identifier] = struct(
            architectures = architectures,
            binary = fat_binary,
            dsym_binaries = dsym_binaries,
            environment = environment,
            linkmaps = linkmaps,
            platform = platform,
            split_attr_keys = split_attr_keys,
            framework_swift_infos = framework_swift_infos,
            uses_swift = uses_swift,
        )

    return link_outputs_by_library_identifier

def _library_identifier(*, architectures, environment, platform):
    """Return a unique identifier for an embedded framework to disambiguate it from others.

    Args:
        architectures: The architectures of the target that was built. For example, `x86_64` or
            `arm64`.
        environment: The environment of the target that was built, which corresponds to the
            toolchain's target triple values as reported by `apple_common` linking APIs.
            Typically `device` or `simulator`.
        platform: The platform of the target that was built, which corresponds to the toolchain's
            target triple values as reported by `apple_common` linking APIs.
            For example, `ios`, `macos`, `tvos`, `visionos` or `watchos`.

    Returns:
        A string that can be used to determine the subfolder this embedded framework will be found
        in the final XCFramework bundle. This mirrors the formatting for subfolders as given by the
        xcodebuild -create-xcframework tool.
    """
    library_identifier = "{}-{}".format(platform, "_".join(architectures))
    if environment != "device":
        library_identifier += "-{}".format(environment)
    return library_identifier

def _unioned_attrs(*, attr_names, split_attr, split_attr_keys):
    """Return a list of attribute values unioned for the given attributes, by split attribute key.

     Args:
        attr_names: The rule attributes to union. Assumed to contain lists of values.
        split_attr: The Starlark interface for 1:2+ transitions, typically from `ctx.split_attr`.
        split_attr_keys: A list of strings representing each 1:2+ transition key to check.

    Returns:
        A new list of attributes based on the union of all rule attributes given, by split
        attribute key.
    """
    unioned_attrs = []
    for attr_name in attr_names:
        attr = getattr(split_attr, attr_name)
        if not attr:
            continue
        for split_attr_key in split_attr_keys:
            found_attr = attr.get(split_attr_key)
            if found_attr:
                unioned_attrs += found_attr
    return unioned_attrs

def _available_library_dictionary(
        *,
        architectures,
        environment,
        headers_path,
        library_identifier,
        library_path,
        platform):
    """Generates a dictionary containing keys referencing a framework in the XCFramework bundle.

     Args:
        architectures: The architectures of the target that was built. For example, `x86_64` or
            `arm64`.
        environment: The environment of the target that was built, which corresponds to the
            toolchain's target triple values as reported by `apple_common` linking APIs.
            Typically `device` or `simulator`.
        headers_path: A string representing the path inside the library identifier to reference
            bundled headers and modulemap files.
        library_identifier: A string representing the path to the framework to reference in the
            xcframework bundle.
        library_path: A string representing the path inside the library identifier to reference in
            the xcframework bundle.
        platform: The platform of the target that was built, which corresponds to the toolchain's
            target triple values as reported by `apple_common` linking APIs.
            For example, `ios`, `macos`, `tvos`, `visionos`, or `watchos`.

    Returns:
        A dictionary containing keys representing how a given framework should be referenced in the
        root Info.plist of a given XCFramework bundle.
    """
    available_library = {
        "LibraryIdentifier": library_identifier,
        "LibraryPath": library_path,
        "SupportedArchitectures": architectures,
        "SupportedPlatform": platform,
    }

    if headers_path:
        available_library["HeadersPath"] = headers_path

    if environment != "device":
        available_library["SupportedPlatformVariant"] = environment
    return available_library

def _create_xcframework_root_infoplist(
        *,
        actions,
        apple_fragment,
        available_libraries,
        plisttool,
        rule_label,
        xcode_config):
    """Generates a root Info.plist for a given XCFramework.

     Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        available_libraries: A dictionary containing keys representing how a given framework should
            be referenced in the root Info.plist of a given XCFramework bundle.
        plisttool: A files_to_run for the plist tool.
        rule_label: The label of the target being analyzed.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.

    Returns:
        A `File` representing a root Info.plist to be embedded within an XCFramework bundle.
    """
    root_info_plist = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = None,
        file_name = "Info.plist",
    )

    default_xcframework_plist = {
        "CFBundlePackageType": "XFWK",
        "XCFrameworkFormatVersion": "1.0",
    }

    plisttool_control = struct(
        binary = False,
        output = root_info_plist.path,
        plists = [{"AvailableLibraries": available_libraries}, default_xcframework_plist],
        target = str(rule_label),
    )
    plisttool_control_file = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = None,
        file_name = "xcframework_plisttool_control.json",
    )
    actions.write(
        output = plisttool_control_file,
        content = json.encode(plisttool_control),
    )
    apple_support.run(
        actions = actions,
        apple_fragment = apple_fragment,
        arguments = [plisttool_control_file.path],
        executable = plisttool,
        inputs = [plisttool_control_file],
        mnemonic = "CreateXCFrameworkRootInfoPlist",
        outputs = [root_info_plist],
        xcode_config = xcode_config,
    )
    return root_info_plist

def _create_xcframework_bundle(
        *,
        actions,
        apple_fragment,
        apple_mac_toolchain_info,
        apple_xplat_toolchain_info,
        bundle_name,
        framework_archive_files,
        framework_archive_merge_files,
        framework_archive_merge_zips = [],
        label_name,
        output_archive,
        root_info_plist,
        tree_artifact_enabled,
        xcode_config):
    """Generates the bundle archive for an XCFramework.

     Args:
        actions: The actions providerx from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        apple_mac_toolchain_info: The `AppleMacToolsToolchainInfo` provider from the mac toolchain.
        apple_xplat_toolchain_info: The `AppleXPlatToolsToolchainInfo` provider from the xplat
            toolchain.
        bundle_name: The name of the XCFramework bundle.
        framework_archive_files: A list of depsets referencing files to be used as inputs to the
            bundling action. This should include every archive referenced as a "src" of
            framework_archive_merge_zips.
        framework_archive_merge_files: A list of structs representing files that should be merged
            into the bundle. Each struct contains two fields: "src", the path of the file that
            should be merged into the bundle; and "dest", the path inside the bundle where the file
            should be placed. The destination path is relative to `bundle_path`.
        framework_archive_merge_zips: A list of structs representing ZIP archives whose contents
            should be merged into the bundle. Each struct contains two fields: "src", the path of
            the archive whose contents should be merged into the bundle; and "dest", the path inside
            the bundle where the ZIPs contents should be placed. The destination path is relative to
            `bundle_path`.
        label_name: Name of the target being built.
        output_archive: The file representing the final bundled archive.
        root_info_plist: A `File` representing a fully formed root Info.plist for this XCFramework.
        tree_artifact_enabled: A boolean indicating whether tree artifact outputs are enabled.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    """
    bundletool_control_file = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = None,
        file_name = "xcframework_bundletool_control.json",
    )
    root_info_plist_merge_file = struct(src = root_info_plist.path, dest = "Info.plist")
    bundletool_control = struct(
        bundle_merge_files = [root_info_plist_merge_file] + framework_archive_merge_files,
        bundle_merge_zips = framework_archive_merge_zips,
        bundle_path = bundle_name + ".xcframework",
        output = output_archive.path,
        compress = True,
    )
    actions.write(
        output = bundletool_control_file,
        content = json.encode(bundletool_control),
    )

    if tree_artifact_enabled:
        bundletool = apple_mac_toolchain_info.bundletool_experimental
        apple_support.run(
            actions = actions,
            apple_fragment = apple_fragment,
            arguments = [bundletool_control_file.path],
            executable = bundletool,
            execution_requirements = {
                # Added so that the output of this action is not cached remotely, in case multiple
                # developers sign the same artifact with different identities.
                "no-remote": "1",
                # Unsure, but may be needed for keychain access, especially for files that live in
                # $HOME.
                "no-sandbox": "1",
            },
            inputs = depset(
                direct = [bundletool_control_file, root_info_plist],
                transitive = framework_archive_files,
            ),
            mnemonic = "CreateXCFrameworkBundle",
            outputs = [output_archive],
            progress_message = "Bundling %s" % label_name,
            xcode_config = xcode_config,
        )
    else:
        bundletool = apple_xplat_toolchain_info.bundletool
        actions.run(
            arguments = [bundletool_control_file.path],
            executable = bundletool,
            inputs = depset(
                direct = [bundletool_control_file, root_info_plist],
                transitive = framework_archive_files,
            ),
            mnemonic = "CreateXCFrameworkBundle",
            outputs = [output_archive],
            progress_message = "Bundling %s" % label_name,
        )

def _apple_xcframework_impl(ctx):
    """Implementation of apple_xcframework."""
    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name = ctx.attr.bundle_name or ctx.attr.name
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = getattr(ctx.attr, "executable_name", bundle_name)
    deps = ctx.split_attr.deps
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    tree_artifact_enabled = False
    outputs_archive = ctx.outputs.archive
    if (apple_xplat_toolchain_info.build_settings.use_tree_artifacts_outputs or
        is_experimental_tree_artifact_enabled(config_vars = ctx.var)):
        actions.write(
            output = ctx.outputs.archive,
            content = "This is a dummy file because tree artifacts are enabled",
        )
        tree_artifact_enabled = True
        outputs_archive = actions.declare_directory(bundle_name + ".xcframework")

    # Add the disable_legacy_signing feature to the list of features
    # TODO(b/72148898): Remove this when dossier based signing becomes the default.
    features = ctx.features
    features.append("disable_legacy_signing")
    label = ctx.label

    # Bundle extension needs to be ".xcframework" for root bundle, but macos/ios/tvos will always
    # be ".framework"
    nested_bundle_extension = ".framework"

    # Similarly, bundle_id is expected to be in terms of the bundle ID for each embedded framework,
    # as this value is not used in the XCFramework's root Info.plist.
    nested_bundle_id = ctx.attr.bundle_id

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        # Frameworks do not have entitlements.
        entitlements = None,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = [
            # iOS, tvOS and watchOS single target app framework binaries live in
            # Application.app/Frameworks/Framework.framework/Framework
            # watchOS 2 extension-dependent app framework binaries live in
            # Application.app/PlugIns/Extension.appex/Frameworks/Framework.framework/Framework
            #
            # iOS, tvOS and watchOS single target app frameworks are packaged in executable as
            # Application.app/Frameworks
            # watchOS 2 extension-dependent app frameworks are packaged in executable as
            # Application.app/PlugIns/Extension.appex/Frameworks
            #
            # While different, these resolve to the same paths relative to their respective
            # executables. Only macOS (which is not yet supported) is an outlier; this will require
            # changes to native Bazel linking logic for Apple binary targets.
            "-Wl,-rpath,@executable_path/Frameworks",
            "-dynamiclib",
            "-Wl,-install_name,@rpath/{name}{extension}/{name}".format(
                extension = nested_bundle_extension,
                name = bundle_name,
            ),
        ] + (["-fapplication-extension"] if ctx.attr.extension_safe else []),
        platform_prerequisites = None,
        # All required knowledge for 3P facing frameworks is passed directly through the given
        # `extra_linkopts`; no rule_descriptor is needed to share with this linking action.
        rule_descriptor = None,
        stamp = ctx.attr.stamp,
    )

    link_outputs_by_library_identifier = _group_link_outputs_by_library_identifier(
        actions = actions,
        apple_fragment = ctx.fragments.apple,
        deps = deps,
        label_name = label.name,
        link_result = link_result,
        xcode_config = xcode_config,
    )

    available_libraries = []
    framework_archive_files = []
    framework_archive_merge_files = []
    framework_archive_merge_zips = []
    framework_output_files = []
    framework_output_groups = []

    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    for library_identifier, link_output in link_outputs_by_library_identifier.items():
        binary_artifact = link_output.binary

        rule_descriptor = rule_support.rule_descriptor(
            platform_type = link_output.platform,
            product_type = apple_product_type.framework,
        )

        platform_prerequisites = platform_support.platform_prerequisites(
            apple_fragment = ctx.fragments.apple,
            build_settings = apple_xplat_toolchain_info.build_settings,
            config_vars = ctx.var,
            cpp_fragment = ctx.fragments.cpp,
            device_families = ctx.attr.families_required.get(
                link_output.platform,
                default = rule_descriptor.allowed_device_families,
            ),
            explicit_minimum_deployment_os = ctx.attr.minimum_deployment_os_versions.get(
                link_output.platform,
            ),
            explicit_minimum_os = ctx.attr.minimum_os_versions.get(link_output.platform),
            features = features,
            objc_fragment = ctx.fragments.objc,
            platform_type_string = link_output.platform,
            uses_swift = link_output.uses_swift,
            xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
        )

        overridden_predeclared_outputs = struct(
            archive = intermediates.file(
                actions = actions,
                target_name = label.name,
                output_discriminator = library_identifier,
                file_name = label.name + ".zip",
            ),
        )

        resource_deps = _unioned_attrs(
            attr_names = ["data", "deps"],
            split_attr = ctx.split_attr,
            split_attr_keys = link_output.split_attr_keys,
        )

        top_level_infoplists = resources.collect(
            attr = ctx.split_attr,
            res_attrs = ["infoplists"],
            split_attr_keys = link_output.split_attr_keys,
        )
        top_level_resources = resources.collect(
            attr = ctx.split_attr,
            res_attrs = ["data"],
            split_attr_keys = link_output.split_attr_keys,
        )

        environment_plist = files.get_file_with_name(
            name = "environment_plist_{platform}".format(
                platform = link_output.platform,
            ),
            files = ctx.files._environment_plist_files,
        )

        processor_partials = [
            partials.apple_bundle_info_partial(
                actions = actions,
                bundle_extension = nested_bundle_extension,
                bundle_id = nested_bundle_id,
                bundle_name = bundle_name,
                entitlements = None,
                executable_name = executable_name,
                extension_safe = ctx.attr.extension_safe,
                label_name = label.name,
                output_discriminator = library_identifier,
                platform_prerequisites = platform_prerequisites,
                predeclared_outputs = overridden_predeclared_outputs,
                product_type = rule_descriptor.product_type,
                rule_descriptor = rule_descriptor,
            ),
            partials.binary_partial(
                actions = actions,
                binary_artifact = binary_artifact,
                bundle_name = bundle_name,
                executable_name = executable_name,
                label_name = label.name,
                output_discriminator = library_identifier,
            ),
            partials.debug_symbols_partial(
                actions = actions,
                bundle_extension = nested_bundle_extension,
                bundle_name = bundle_name,
                debug_discriminator = link_output.platform + "_" + link_output.environment,
                dsym_binaries = link_output.dsym_binaries,
                dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
                executable_name = executable_name,
                label_name = label.name,
                linkmaps = link_output.linkmaps,
                output_discriminator = library_identifier,
                platform_prerequisites = platform_prerequisites,
                plisttool = apple_mac_toolchain_info.plisttool,
                rule_label = label,
                version = ctx.attr.version,
            ),
            partials.resources_partial(
                actions = actions,
                apple_mac_toolchain_info = apple_mac_toolchain_info,
                bundle_extension = nested_bundle_extension,
                bundle_id = nested_bundle_id,
                bundle_name = bundle_name,
                environment_plist = environment_plist,
                executable_name = executable_name,
                launch_storyboard = None,
                output_discriminator = library_identifier,
                platform_prerequisites = platform_prerequisites,
                resource_deps = resource_deps,
                rule_descriptor = rule_descriptor,
                rule_label = label,
                top_level_infoplists = top_level_infoplists,
                top_level_resources = top_level_resources,
                version = ctx.attr.version,
                version_keys_required = False,
            ),
            partials.swift_dylibs_partial(
                actions = actions,
                apple_mac_toolchain_info = apple_mac_toolchain_info,
                binary_artifact = binary_artifact,
                label_name = label.name,
                platform_prerequisites = platform_prerequisites,
            ),
        ]

        if link_output.framework_swift_infos:
            processor_partials.append(
                partials.swift_framework_partial(
                    actions = actions,
                    bundle_name = bundle_name,
                    label_name = label.name,
                    output_discriminator = library_identifier,
                    swift_infos = link_output.framework_swift_infos,
                ),
            )
        else:
            processor_partials.append(
                partials.framework_header_modulemap_partial(
                    actions = actions,
                    bundle_name = bundle_name,
                    hdrs = ctx.files.public_hdrs,
                    label_name = label.name,
                    output_discriminator = library_identifier,
                    umbrella_header = ctx.file.umbrella_header,
                ),
            )

        processor_result = processor.process(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = nested_bundle_extension,
            bundle_name = bundle_name,
            entitlements = None,
            features = features,
            ipa_post_processor = None,
            output_discriminator = library_identifier,
            partials = processor_partials,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = overridden_predeclared_outputs,
            process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
            provisioning_profile = None,
            rule_descriptor = rule_descriptor,
            rule_label = label,
        )

        for provider in processor_result.providers:
            # Save the framework archive.
            if getattr(provider, "archive", None):
                # Repackage every archive found for bundle_merge_files or bundle_merge_zips in the
                # final bundler action, depending on whether tree artifacts are enabled.
                if tree_artifact_enabled:
                    framework_archive_merge_files.append(
                        struct(
                            src = provider.archive.path,
                            dest = paths.join(
                                library_identifier,
                                bundle_name + nested_bundle_extension,
                            ),
                        ),
                    )
                else:
                    framework_archive_merge_zips.append(
                        struct(
                            src = provider.archive.path,
                            dest = library_identifier,
                        ),
                    )

                # Save a reference to those archives as file-friendly inputs to the bundler action.
                framework_archive_files.append(depset([provider.archive]))

            # Save the dSYMs.
            if getattr(provider, "dsyms", None):
                framework_output_files.append(depset(transitive = [provider.dsyms]))
                framework_output_groups.append({"dsyms": provider.dsyms})

            # Save the linkmaps.
            if getattr(provider, "linkmaps", None):
                framework_output_files.append(depset(transitive = [provider.linkmaps]))
                framework_output_groups.append({"linkmaps": provider.linkmaps})

        # Save additional library details for the XCFramework's root info plist.
        available_libraries.append(_available_library_dictionary(
            architectures = link_output.architectures,
            environment = link_output.environment,
            headers_path = None,
            library_identifier = library_identifier,
            library_path = bundle_name + nested_bundle_extension,
            platform = link_output.platform,
        ))

    root_info_plist = _create_xcframework_root_infoplist(
        actions = actions,
        apple_fragment = ctx.fragments.apple,
        available_libraries = available_libraries,
        plisttool = apple_mac_toolchain_info.plisttool,
        rule_label = label,
        xcode_config = xcode_config,
    )

    _create_xcframework_bundle(
        actions = actions,
        apple_fragment = ctx.fragments.apple,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_name = bundle_name,
        framework_archive_files = framework_archive_files,
        framework_archive_merge_files = framework_archive_merge_files,
        framework_archive_merge_zips = framework_archive_merge_zips,
        label_name = label.name,
        output_archive = outputs_archive,
        root_info_plist = root_info_plist,
        tree_artifact_enabled = tree_artifact_enabled,
        xcode_config = xcode_config,
    )

    processor_output = [
        # Limiting the contents of AppleBundleInfo to what is necessary for testing and validation.
        new_applebundleinfo(
            archive = outputs_archive,
            bundle_extension = ".xcframework",
            bundle_id = nested_bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            infoplist = root_info_plist,
            platform_type = None,
        ),
        new_applexcframeworkbundleinfo(),
        DefaultInfo(
            files = depset([outputs_archive], transitive = framework_output_files),
        ),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                *framework_output_groups
            )
        ),
    ]
    return processor_output

apple_xcframework = rule_factory.create_apple_rule(
    cfg = None,
    doc = "Builds and bundles an XCFramework for third-party distribution.",
    implementation = _apple_xcframework_impl,
    predeclared_outputs = {"archive": "%{name}.xcframework.zip"},
    toolchains = [],
    attrs = [
        rule_attrs.common_tool_attrs(),
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.xcframework_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = False,
        ),
        {
            "_environment_plist_files": attr.label_list(
                default = [
                    "//apple/internal:environment_plist_ios",
                    "//apple/internal:environment_plist_tvos",
                ],
            ),
            "bundle_id": attr.string(
                mandatory = True,
                doc = """
The bundle ID (reverse-DNS path followed by app name) for each of the embedded frameworks. If
present, this value will be embedded in an Info.plist within each framework bundle.
""",
            ),
            "bundle_name": attr.string(
                mandatory = False,
                doc = """
The desired name of the xcframework bundle (without the extension) and the bundles for all embedded
frameworks. If this attribute is not set, then the name of the target will be used instead.
""",
            ),
            "data": attr.label_list(
                allow_files = True,
                aspects = [apple_resource_aspect],
                cfg = transition_support.xcframework_transition,
                doc = """
A list of resources or files bundled with the bundle. The resources will be stored in the
appropriate resources location within each of the embedded framework bundles.
""",
            ),
            "extension_safe": attr.bool(
                default = False,
                doc = """
If true, compiles and links this framework with `-application-extension`, restricting the binary to
use only extension-safe APIs.
""",
            ),
            "families_required": attr.string_list_dict(
                doc = """
A list of device families supported by this extension, with platforms such as `ios` as keys. Valid
values are `iphone` and `ipad` for `ios`; at least one must be specified if a platform is defined.
Currently, this only affects processing of `ios` resources.
""",
            ),
            "infoplists": attr.label_list(
                allow_empty = False,
                allow_files = [".plist"],
                cfg = transition_support.xcframework_transition,
                doc = """
A list of .plist files that will be merged to form the Info.plist for each of the embedded
frameworks. At least one file must be specified. Please see
[Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling)
for what is supported.
""",
                mandatory = True,
            ),
            "ios": attr.string_list_dict(
                doc = """
A dictionary of strings indicating which platform variants should be built for the iOS platform (
`device` or `simulator`) as keys, and arrays of strings listing which architectures should be
built for those platform variants (for example, `x86_64`, `arm64`) as their values.
""",
            ),
            "macos": attr.string_list(
                doc = """
A list of strings indicating which architecture should be built for the macOS platform (for example, `x86_64`, `arm64`).
""",
            ),
            "tvos": attr.string_list_dict(
                doc = """
A dictionary of strings indicating which platform variants should be built for the tvOS platform (
`device` or `simulator`) as keys, and arrays of strings listing which architectures should be
built for those platform variants (for example, `x86_64`, `arm64`) as their values.
""",
            ),
            "minimum_deployment_os_versions": attr.string_dict(
                doc = """
A dictionary of strings indicating the minimum deployment OS version supported by the target,
represented as a dotted version number (for example, "9.0") as values, with their respective
platforms such as `ios` as keys. This is different from `minimum_os_versions`, which is effective
at compile time. Ensure version specific APIs are guarded with `available` clauses.
""",
                mandatory = False,
            ),
            "minimum_os_versions": attr.string_dict(
                doc = """
A dictionary of strings indicating the minimum OS version supported by the target, represented as a
dotted version number (for example, "8.0") as values, with their respective platforms such as `ios`,
or `tvos` as keys:

    minimum_os_versions = {
        "ios": "13.0",
        "tvos": "15.0",
    }
""",
                mandatory = True,
            ),
            "public_hdrs": attr.label_list(
                allow_files = [".h"],
                doc = """
A list of files directly referencing header files to be used as the publicly visible interface for
each of these embedded frameworks. These header files will be embedded within each bundle,
typically in a subdirectory such as `Headers`.
""",
            ),
            "version": attr.label(
                providers = [[AppleBundleVersionInfo]],
                doc = """
An `apple_bundle_version` target that represents the version for this target. See
[`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-versioning.md#apple_bundle_version).
""",
            ),
            "umbrella_header": attr.label(
                allow_single_file = [".h"],
                doc = """
An optional single .h file to use as the umbrella header for this framework. Usually, this header
will have the same name as this target, so that clients can load the header using the #import
<MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an
umbrella header will be generated under the same name as this target.
""",
            ),
        },
    ],
)

def _apple_static_xcframework_impl(ctx):
    """Implementation of apple_static_xcframework."""

    actions = ctx.actions
    apple_fragment = ctx.fragments.apple
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name = ctx.attr.bundle_name or ctx.label.name
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    deps = ctx.split_attr.deps
    label = ctx.label
    executable_name = getattr(ctx.attr, "executable_name", bundle_name)
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    tree_artifact_enabled = False
    outputs_archive = ctx.outputs.archive
    if (apple_xplat_toolchain_info.build_settings.use_tree_artifacts_outputs or
        is_experimental_tree_artifact_enabled(config_vars = ctx.var)):
        actions.write(
            output = ctx.outputs.archive,
            content = "This is a dummy file because tree artifacts are enabled",
        )
        tree_artifact_enabled = True
        outputs_archive = actions.declare_directory(bundle_name + ".xcframework")

    archive_result = linking_support.register_static_library_archive_action(
        ctx = ctx,
        cc_toolchains = cc_toolchain_forwarder,
    )
    link_outputs_by_library_identifier = _group_link_outputs_by_library_identifier(
        actions = actions,
        apple_fragment = apple_fragment,
        deps = deps,
        label_name = bundle_name,
        link_result = archive_result,
        xcode_config = xcode_config,
    )

    available_libraries = []
    framework_archive_files = []
    framework_archive_merge_files = []
    for library_identifier, link_output in link_outputs_by_library_identifier.items():
        # Bundle binary artifact for specific library identifier
        binary_artifact = link_output.binary
        framework_dir = paths.join(library_identifier, bundle_name + ".framework")
        framework_archive_merge_files.append(struct(
            src = binary_artifact.path,
            dest = paths.join(framework_dir, bundle_name),
        ))
        framework_archive_files.append(depset([binary_artifact]))

        if link_output.framework_swift_infos:
            # Generate headers, modulemaps, and swiftmodules
            interface_artifacts = partial.call(
                partials.swift_framework_partial(
                    actions = actions,
                    avoid_deps = ctx.attr.avoid_deps,
                    bundle_name = bundle_name,
                    framework_modulemap = True,
                    label_name = label.name,
                    output_discriminator = library_identifier,
                    swift_infos = link_output.framework_swift_infos,
                ),
            )
        else:
            # Generate headers, and modulemaps
            sdk_frameworks = cc_info_support.get_sdk_frameworks(
                deps = ctx.split_attr.deps,
                split_deps_keys = link_output.split_attr_keys,
            )
            sdk_dylibs = cc_info_support.get_sdk_dylibs(
                deps = ctx.split_attr.deps,
                split_deps_keys = link_output.split_attr_keys,
            )
            interface_artifacts = partial.call(partials.framework_header_modulemap_partial(
                actions = actions,
                bundle_name = bundle_name,
                framework_modulemap = True,
                hdrs = ctx.files.public_hdrs,
                label_name = label.name,
                output_discriminator = library_identifier,
                umbrella_header = ctx.file.umbrella_header,
                sdk_frameworks = sdk_frameworks,
                sdk_dylibs = sdk_dylibs,
            ))

        # Bundle headers & modulemaps (and swiftmodules if available)
        for _, bundle_relative_path, files in interface_artifacts.bundle_files:
            framework_archive_files.append(files)
            for file in files.to_list():
                framework_archive_merge_files.append(struct(
                    src = file.path,
                    dest = paths.join(
                        framework_dir,
                        bundle_relative_path,
                        file.basename,
                    ),
                ))

        # Save additional library details for the XCFramework's root info plist.
        available_libraries.append(
            _available_library_dictionary(
                architectures = link_output.architectures,
                environment = link_output.environment,
                headers_path = None,
                library_identifier = library_identifier,
                library_path = bundle_name + ".framework",
                platform = link_output.platform,
            ),
        )

        # Bundle resources
        rule_descriptor = rule_support.rule_descriptor(
            platform_type = link_output.platform,
            product_type = apple_product_type.framework,
        )
        platform_prerequisites = platform_support.platform_prerequisites(
            apple_fragment = ctx.fragments.apple,
            build_settings = apple_xplat_toolchain_info.build_settings,
            config_vars = ctx.var,
            cpp_fragment = ctx.fragments.cpp,
            device_families = ctx.attr.families_required.get(
                link_output.platform,
                default = rule_descriptor.allowed_device_families,
            ),
            explicit_minimum_deployment_os = ctx.attr.minimum_deployment_os_versions.get(
                link_output.platform,
            ),
            explicit_minimum_os = ctx.attr.minimum_os_versions.get(link_output.platform),
            features = features,
            objc_fragment = ctx.fragments.objc,
            platform_type_string = link_output.platform,
            uses_swift = link_output.uses_swift,
            xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
        )
        resource_deps = _unioned_attrs(
            attr_names = ["deps"],
            split_attr = ctx.split_attr,
            split_attr_keys = link_output.split_attr_keys,
        )
        targets_to_avoid = _unioned_attrs(
            attr_names = ["avoid_deps"],
            split_attr = ctx.split_attr,
            split_attr_keys = link_output.split_attr_keys,
        )

        # Collect top_level_infoplists only if bundle_id is set
        top_level_infoplists = []
        if ctx.attr.bundle_id:
            top_level_infoplists = resources.collect(
                attr = ctx.split_attr,
                res_attrs = ["infoplists"],
                split_attr_keys = link_output.split_attr_keys,
            )

        partial_output = partial.call(partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = ".framework",
            bundle_id = ctx.attr.bundle_id,
            bundle_name = bundle_name,
            # TODO(b/174858377): Select which environment_plist to use based on Apple platform.
            environment_plist = ctx.file._environment_plist_ios,
            executable_name = executable_name,
            launch_storyboard = None,
            output_discriminator = library_identifier,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            targets_to_avoid = targets_to_avoid,
            top_level_infoplists = top_level_infoplists,
            version = ctx.attr.version,
        ))

        if getattr(partial_output, "bundle_files", None):
            for target_location, parent, sources in partial_output.bundle_files:
                parent_output_directory = parent or ""
                if target_location != "resource" and target_location != "content":
                    # If we need to add more in the future we should be sure to
                    # double check where the need to end up
                    fail("Got unexpected target location '{}' for '{}'"
                        .format(target_location, sources.to_list()))

                framework_archive_files.append(sources)
                for source in sources.to_list():
                    target_path = parent_output_directory
                    if not source.is_directory:
                        target_path = paths.join(target_path, source.basename)

                    framework_archive_merge_files.append(struct(
                        src = source.path,
                        dest = paths.join(
                            framework_dir,
                            target_path,
                        ),
                    ))

    root_info_plist = _create_xcframework_root_infoplist(
        actions = actions,
        apple_fragment = apple_fragment,
        available_libraries = available_libraries,
        plisttool = apple_mac_toolchain_info.plisttool,
        rule_label = label,
        xcode_config = xcode_config,
    )

    _create_xcframework_bundle(
        actions = actions,
        apple_fragment = apple_fragment,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_name = bundle_name,
        framework_archive_files = framework_archive_files,
        framework_archive_merge_files = framework_archive_merge_files,
        label_name = label.name,
        output_archive = outputs_archive,
        root_info_plist = root_info_plist,
        tree_artifact_enabled = tree_artifact_enabled,
        xcode_config = xcode_config,
    )

    return [
        # Limiting the contents of AppleBundleInfo to what is necessary for testing and validation.
        new_applebundleinfo(
            archive = outputs_archive,
            bundle_extension = ".xcframework",
            bundle_name = bundle_name,
            executable_name = executable_name,
            infoplist = root_info_plist,
            platform_type = None,
        ),
        new_applestaticxcframeworkbundleinfo(),
        DefaultInfo(
            files = depset([outputs_archive]),
        ),
    ]

apple_static_xcframework = rule_factory.create_apple_rule(
    cfg = None,
    doc = "Generates an XCFramework with static libraries for third-party distribution.",
    implementation = _apple_static_xcframework_impl,
    predeclared_outputs = {"archive": "%{name}.xcframework.zip"},
    toolchains = [],
    attrs = [
        rule_attrs.common_tool_attrs(),
        rule_attrs.static_library_archive_attrs(
            deps_cfg = transition_support.xcframework_transition,
        ),
        {
            "bundle_id": attr.string(
                mandatory = False,
                doc = """
Optional bundle ID (reverse-DNS path followed by framework name) for each of the embedded frameworks.
If present, this value will be embedded in an Info.plist within each framework bundle, similar to
apple_xcframework (dynamic frameworks).
""",
            ),
            "executable_name": attr.string(
                mandatory = False,
                doc = """
The desired name of the executable, if the bundle has an executable. If this attribute is not set,
then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of
the target will be used instead.
""",
            ),
            "_environment_plist_ios": attr.label(
                allow_single_file = True,
                default = "//apple/internal:environment_plist_ios",
            ),
            "avoid_deps": attr.label_list(
                aspects = [apple_resource_aspect],
                allow_files = True,
                cfg = transition_support.xcframework_transition,
                mandatory = False,
                doc = """
A list of library targets on which this framework depends in order to compile, but the transitive
closure of which will not be linked into the framework's binary, nor bundled into final XCFramework.
""",
            ),
            "bundle_name": attr.string(
                mandatory = False,
                doc = """
The desired name of the XCFramework bundle (without the extension) and the binaries for all embedded
static libraries. If this attribute is not set, then the name of the target will be used instead.
""",
            ),
            "deps": attr.label_list(
                aspects = [apple_resource_aspect, swift_usage_aspect],
                allow_files = True,
                cfg = transition_support.xcframework_transition,
                mandatory = True,
                doc = """
A list of files directly referencing libraries to be represented for each given platform split in
the XCFramework. These libraries will be embedded within each platform split.
""",
            ),
            "families_required": attr.string_list_dict(
                doc = """
A list of device families supported by this extension, with platforms such as `ios` as keys. Valid
values are `iphone` and `ipad` for `ios`; at least one must be specified if a platform is defined.
Currently, this only affects processing of `ios` resources.
""",
            ),
            "infoplists": attr.label_list(
                allow_empty = True,
                allow_files = [".plist"],
                cfg = transition_support.xcframework_transition,
                doc = """
A list of .plist files that will be merged to form the Info.plist for each of the embedded
frameworks. Only used if bundle_id is provided.
""",
                mandatory = False,
            ),
            "ios": attr.string_list_dict(
                doc = """
A dictionary of strings indicating which platform variants should be built for the `ios` platform (
`device` or `simulator`) as keys, and arrays of strings listing which architectures should be
built for those platform variants (for example, `x86_64`, `arm64`) as their values.
""",
            ),
            "macos": attr.string_list(
                doc = """
A list of strings indicating which architecture should be built for the macOS platform (for example, `x86_64`, `arm64`).
""",
            ),
            "minimum_deployment_os_versions": attr.string_dict(
                doc = """
A dictionary of strings indicating the minimum deployment OS version supported by the target,
represented as a dotted version number (for example, "9.0") as values, with their respective
platforms such as `ios` as keys. This is different from `minimum_os_versions`, which is effective
at compile time. Ensure version specific APIs are guarded with `available` clauses.
""",
                mandatory = False,
            ),
            "minimum_os_versions": attr.string_dict(
                mandatory = True,
                doc = """
A dictionary of strings indicating the minimum OS version supported by the target, represented as a
dotted version number (for example, "8.0") as values, with their respective platforms such as `ios`
as keys.
""",
            ),
            "public_hdrs": attr.label_list(
                allow_files = [".h"],
                cfg = transition_support.xcframework_transition,
                doc = """
A list of files directly referencing header files to be used as the publicly visible interface for
each of these embedded libraries. These header files will be embedded within each platform split,
typically in a subdirectory such as `Headers`.
""",
            ),
            "umbrella_header": attr.label(
                allow_single_file = [".h"],
                doc = """
An optional single .h file to use as the umbrella header for this framework. Usually, this header
will have the same name as this target, so that clients can load the header using the #import
<MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an
umbrella header will be generated under the same name as this target.
""",
            ),
            "version": attr.label(
                providers = [[AppleBundleVersionInfo]],
                doc = """
An `apple_bundle_version` target that represents the version for this target. See
[`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-versioning.md#apple_bundle_version).
""",
            ),
        },
    ],
)
