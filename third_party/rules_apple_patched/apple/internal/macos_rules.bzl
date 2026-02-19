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

"""Implementation of macOS rules."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple:providers.bzl",
    "AppleBinaryInfoplistInfo",
    "AppleBundleInfo",
    "AppleBundleVersionInfo",
    "ApplePlatformInfo",
    "MacosExtensionBundleInfo",
    "MacosFrameworkBundleInfo",
    "MacosStaticFrameworkBundleInfo",
    "MacosXPCServiceBundleInfo",
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
    "bundle_id_suffix_default",
    "bundling_support",
)
load(
    "//apple/internal:cc_info_support.bzl",
    "cc_info_support",
)
load(
    "//apple/internal:codesigning_support.bzl",
    "codesigning_support",
)
load(
    "//apple/internal:entitlements_support.bzl",
    "entitlements_support",
)
load(
    "//apple/internal:features_support.bzl",
    "features_support",
)
load(
    "//apple/internal:framework_import_support.bzl",
    "libraries_to_link_for_dynamic_framework",
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
    "AppleExecutableBinaryInfo",
    "new_applebinaryinfo",
    "new_appleexecutablebinaryinfo",
    "new_appleframeworkbundleinfo",
    "new_macosapplicationbundleinfo",
    "new_macosbundlebundleinfo",
    "new_macosextensionbundleinfo",
    "new_macoskernelextensionbundleinfo",
    "new_macosquicklookpluginbundleinfo",
    "new_macosspotlightimporterbundleinfo",
    "new_macosxpcservicebundleinfo",
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
    "//apple/internal:run_support.bzl",
    "run_support",
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
    "//apple/internal/aspects:framework_provider_aspect.bzl",
    "framework_provider_aspect",
)
load(
    "//apple/internal/aspects:resource_aspect.bzl",
    "apple_resource_aspect",
)
load(
    "//apple/internal/aspects:swift_dynamic_framework_aspect.bzl",
    "SwiftDynamicFrameworkInfo",
    "swift_dynamic_framework_aspect",
)
load(
    "//apple/internal/utils:clang_rt_dylibs.bzl",
    "clang_rt_dylibs",
)
load(
    "//apple/internal/utils:main_thread_checker_dylibs.bzl",
    "main_thread_checker_dylibs",
)

def _macos_application_impl(ctx):
    """Implementation of macos_application."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.application,
    )

    embedded_targets = (
        ctx.attr.frameworks +
        ctx.attr.extensions +
        ctx.attr.xpc_services +
        ctx.attr.deps
    )
    verification_targets = (
        ctx.attr.frameworks +
        ctx.attr.extensions +
        ctx.attr.xpc_services
    )
    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    bundle_verification_targets = [struct(target = ext) for ext in verification_targets]
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = [
            "app_icons",
            "strings",
            "resources",
        ],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        avoid_deps = ctx.attr.frameworks,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    processor_partials = [
        partials.app_intents_metadata_bundle_partial(
            actions = actions,
            cc_toolchains = cc_toolchain_forwarder,
            ctx = ctx,
            deps = ctx.split_attr.app_intents,
            disabled_features = ctx.disabled_features,
            features = features,
            label = label,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            entitlements = entitlements.bundle,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            embedded_targets = embedded_targets,
            entitlements = entitlements.codesigning,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = embedded_targets + ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            bundle_embedded_bundles = True,
            embeddable_targets = embedded_targets,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.cc_info_dylibs_partial(
            embedded_targets = embedded_targets,
        ),
        partials.framework_import_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
            targets = embedded_targets,
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            bundle_verification_targets = bundle_verification_targets,
            environment_plist = ctx.file._environment_plist,
            launch_storyboard = None,
            locales_to_include = ctx.attr.locales_to_include,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            targets_to_avoid = ctx.attr.frameworks,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            bundle_dylibs = True,
            dependency_targets = embedded_targets,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = embedded_targets,
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = ctx.attr.include_symbols_in_bundle,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        locales_to_include = ctx.attr.locales_to_include,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    executable = outputs.executable(
        actions = actions,
        label_name = label.name,
    )

    archive = outputs.archive(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )
    dsyms = outputs.dsyms(processor_result = processor_result)

    run_support.register_macos_executable(
        actions = actions,
        archive = archive,
        bundle_name = bundle_name,
        output = executable,
        runner_template = ctx.file._runner_template,
    )

    return [
        coverage_common.instrumented_files_info(ctx, dependency_attributes = ["deps"]),
        DefaultInfo(
            executable = executable,
            files = processor_result.output_files,
            runfiles = ctx.runfiles(
                files = [archive],
                transitive_files = dsyms,
            ),
        ),
        new_macosapplicationbundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        new_appleexecutablebinaryinfo(
            binary = binary_artifact,
            cc_info = link_result.cc_info,
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_bundle_impl(ctx):
    """Implementation of macos_bundle."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.bundle,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_extension = ctx.attr.bundle_extension,
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = [
            "app_icons",
            "strings",
            "resources",
        ],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        bundle_loader = ctx.attr.bundle_loader,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = ["-bundle"],
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive = outputs.archive(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            entitlements = entitlements.bundle,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.plugin,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            platform_prerequisites = platform_prerequisites,
            plugins = [archive],
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(
            files = processor_result.output_files,
        ),
        new_macosbundlebundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_extension_impl(ctx):
    """Implementation of macos_extension."""

    product_type = apple_product_type.app_extension
    if ctx.attr.extensionkit_extension:
        product_type = apple_product_type.extensionkit_extension

    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = product_type,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = [
            "app_icons",
            "strings",
            "resources",
        ],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    extra_linkopts = [
        "-fapplication-extension",
        "-e",
        "_NSExtensionMain",
    ]

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        avoid_deps = ctx.attr.frameworks,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = extra_linkopts,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )

    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive = outputs.archive(
        actions = actions,
        bundle_name = bundle_name,
        bundle_extension = bundle_extension,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )

    embedded_bundles_args = {}
    if rule_descriptor.product_type == apple_product_type.app_extension:
        embedded_bundles_args["plugins"] = [archive]
    elif rule_descriptor.product_type == apple_product_type.extensionkit_extension:
        embedded_bundles_args["extensions"] = [archive]
    else:
        fail("Internal Error: Unexpectedly found product_type " + rule_descriptor.product_type)

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            entitlements = entitlements.bundle,
            executable_name = executable_name,
            extension_safe = True,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.plugin,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            entitlements = entitlements.codesigning,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.frameworks + ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            embeddable_targets = ctx.attr.frameworks,
            platform_prerequisites = platform_prerequisites,
            **embedded_bundles_args
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            extensionkit_keys_required = ctx.attr.extensionkit_extension,
            launch_storyboard = None,
            locales_to_include = ctx.attr.locales_to_include,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            dependency_targets = ctx.attr.frameworks,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = [],
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = False,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        locales_to_include = ctx.attr.locales_to_include,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(
            files = processor_result.output_files,
        ),
        new_appleexecutablebinaryinfo(
            binary = binary_artifact,
            cc_info = link_result.cc_info,
        ),
        new_macosextensionbundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_quick_look_plugin_impl(ctx):
    """Experimental implementation of macos_quick_look_plugin."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.quicklook_plugin,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = [
            "strings",
            "resources",
        ],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = ["-bundle"],
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive = outputs.archive(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            entitlements = entitlements.bundle,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        # TODO(kaipi): Check if clang_rt dylibs are needed in Quick Look plugins, or if
        # they can be skipped.
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.framework,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            entitlements = entitlements.codesigning,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            frameworks = [archive],
            platform_prerequisites = platform_prerequisites,
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = [],
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = False,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(files = processor_result.output_files),
        new_macosquicklookpluginbundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_kernel_extension_impl(ctx):
    """Implementation of macos_kernel_extension."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.kernel_extension,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = ["resources"],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    # This was added for b/122473338, and should be removed eventually once symbol stripping is
    # better-handled. It's redundant with an option added in the CROSSTOOL for the
    # "kernel_extension" feature, but for now it's necessary to detect kext linking so
    # CompilationSupport.java can apply the correct type of symbol stripping.
    extra_linkopts = [
        "-Wl,-kext",
    ]

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = extra_linkopts,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive = outputs.archive(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            entitlements = entitlements.bundle,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.plugin,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            entitlements = entitlements.codesigning,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            platform_prerequisites = platform_prerequisites,
            plugins = [archive],
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = [],
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = False,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(
            files = processor_result.output_files,
        ),
        new_macoskernelextensionbundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_spotlight_importer_impl(ctx):
    """Implementation of macos_spotlight_importer."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.spotlight_importer,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive = outputs.archive(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            entitlements = entitlements.bundle,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.plugin,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            entitlements = entitlements.codesigning,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            platform_prerequisites = platform_prerequisites,
            plugins = [archive],
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            top_level_infoplists = top_level_infoplists,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = [],
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = False,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(
            files = processor_result.output_files,
        ),
        new_macosspotlightimporterbundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_xpc_service_impl(ctx):
    """Implementation of macos_xpc_service."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.xpc_service,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    bundle_id = bundling_support.bundle_full_id(
        bundle_id = ctx.attr.bundle_id,
        bundle_id_suffix = ctx.attr.bundle_id_suffix,
        bundle_name = bundle_name,
        suffix_default = ctx.attr._bundle_id_suffix_default,
        shared_capabilities = ctx.attr.shared_capabilities,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )

    entitlements = entitlements_support.process_entitlements(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        bundle_id = bundle_id,
        entitlements_file = ctx.file.entitlements,
        platform_prerequisites = platform_prerequisites,
        product_type = rule_descriptor.product_type,
        provisioning_profile = provisioning_profile,
        rule_label = label,
        validation_mode = ctx.attr.entitlements_validation,
    )

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        entitlements = entitlements.linking,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive = outputs.archive(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        label_name = label.name,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        rule_descriptor = rule_descriptor,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            executable_name = executable_name,
            bundle_id = bundle_id,
            entitlements = entitlements.bundle,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.xpc_service,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            entitlements = entitlements.codesigning,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.additional_contents.keys(),
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            platform_prerequisites = platform_prerequisites,
            xpc_services = [archive],
        ),
        partials.macos_additional_contents_partial(
            additional_contents = ctx.attr.additional_contents,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            top_level_infoplists = top_level_infoplists,
            version = ctx.attr.version,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = [],
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = False,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if provisioning_profile:
        processor_partials.append(
            partials.provisioning_profile_partial(
                actions = actions,
                extension = "provisionprofile",
                location = processor.location.content,
                profile_artifact = provisioning_profile,
                rule_label = label,
            ),
        )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        entitlements = entitlements.codesigning,
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(
            files = processor_result.output_files,
        ),
        new_macosxpcservicebundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_command_line_application_impl(ctx):
    """Implementation of the macos_command_line_application rule."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.tool,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = None,  # macos_command_line_application doesn't support this override.
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        # Command-line applications do not have entitlements.
        entitlements = None,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    debug_outputs_partial = partials.debug_symbols_partial(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        dsym_binaries = debug_outputs.dsym_binaries,
        dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
        label_name = label.name,
        linkmaps = debug_outputs.linkmaps,
        platform_prerequisites = platform_prerequisites,
        plisttool = apple_mac_toolchain_info.plisttool,
        rule_label = label,
        version = ctx.attr.version,
    )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        bundle_post_process_and_sign = False,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        features = features,
        partials = [debug_outputs_partial],
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )
    output_file = actions.declare_file(label.name)
    codesigning_support.sign_binary_action(
        actions = actions,
        codesign_inputs = ctx.files.codesign_inputs,
        codesigningtool = apple_mac_toolchain_info.codesigningtool,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        input_binary = binary_artifact,
        output_binary = output_file,
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
    )

    infoplists = [
        x[AppleBinaryInfoplistInfo].infoplist
        for x in getattr(ctx.attr, "deps", [])
        if AppleBinaryInfoplistInfo in x
    ]

    # There should only be one `AppleBinaryInfoplistInfo` providing dep
    infoplist = infoplists[0] if infoplists else None

    runfiles = []
    if clang_rt_dylibs.should_package_clang_runtime(features = features):
        runfiles = clang_rt_dylibs.get_from_toolchain(ctx)

    dsyms = outputs.dsyms(processor_result = processor_result)

    return [
        new_applebinaryinfo(
            binary = output_file,
            infoplist = infoplist,
            product_type = rule_descriptor.product_type,
        ),
        DefaultInfo(
            executable = output_file,
            files = depset(transitive = [
                depset([output_file]),
                processor_result.output_files,
            ]),
            runfiles = ctx.runfiles(
                files = runfiles,
                transitive_files = dsyms,
            ),
        ),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        new_appleexecutablebinaryinfo(
            binary = binary_artifact,
            cc_info = link_result.cc_info,
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_dylib_impl(ctx):
    """Implementation of the macos_dylib rule."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.dylib,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = None,  # macos_dylib doesn't support this override.
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        # Dynamic libraries do not have entitlements.
        entitlements = None,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = ["-dynamiclib"],
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    debug_outputs_partial = partials.debug_symbols_partial(
        actions = actions,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        dsym_binaries = debug_outputs.dsym_binaries,
        dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
        label_name = label.name,
        linkmaps = debug_outputs.linkmaps,
        platform_prerequisites = platform_prerequisites,
        plisttool = apple_mac_toolchain_info.plisttool,
        rule_label = label,
        version = ctx.attr.version,
    )

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        bundle_post_process_and_sign = False,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        features = features,
        ipa_post_processor = None,
        partials = [debug_outputs_partial],
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )
    output_file = actions.declare_file(label.name + ".dylib")
    codesigning_support.sign_binary_action(
        actions = actions,
        codesign_inputs = ctx.files.codesign_inputs,
        codesigningtool = apple_mac_toolchain_info.codesigningtool,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        input_binary = binary_artifact,
        output_binary = output_file,
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
    )

    infoplists = [
        x[AppleBinaryInfoplistInfo].infoplist
        for x in getattr(ctx.attr, "deps", [])
        if AppleBinaryInfoplistInfo in x
    ]

    # There should only be one `AppleBinaryInfoplistInfo` providing dep
    infoplist = infoplists[0] if infoplists else None

    return [
        new_applebinaryinfo(
            binary = output_file,
            infoplist = infoplist,
            product_type = rule_descriptor.product_type,
        ),
        DefaultInfo(files = depset(transitive = [
            depset([output_file]),
            processor_result.output_files,
        ])),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
                {"dylib": depset(direct = [output_file])},
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

macos_application = rule_factory.create_apple_rule(
    doc = """Builds and bundles a macOS Application.

This rule creates an application that is a `.app` bundle. If you want to build a
simple command line tool as a standalone binary, use
[`macos_command_line_application`](#macos_command_line_application) instead.""",
    implementation = _macos_application_impl,
    is_executable = True,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.app_icon_attrs(),
        rule_attrs.app_intents_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.locales_to_include_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
            "extensions": attr.label_list(
                providers = [
                    [AppleBundleInfo, MacosExtensionBundleInfo],
                ],
                doc = "A list of macOS extensions to include in the final application bundle.",
            ),
            "frameworks": attr.label_list(
                aspects = [framework_provider_aspect],
                providers = [[AppleBundleInfo, MacosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework))
that this target depends on.
""",
            ),
            "xpc_services": attr.label_list(
                providers = [
                    [AppleBundleInfo, MacosXPCServiceBundleInfo],
                ],
                doc = "A list of macOS XPC Services to include in the final application bundle.",
            ),
            "_runner_template": attr.label(
                cfg = "exec",
                allow_single_file = True,
                default = Label("//apple/internal/templates:macos_template"),
            ),
            "include_symbols_in_bundle": attr.bool(
                default = False,
                doc = """
If true and --output_groups=+dsyms is specified, generates `$UUID.symbols` files from all
`{binary: .dSYM, ...}` pairs for the application and its dependencies, then packages them under the
`Symbols/` directory in the final application bundle.
""",
            ),
        },
    ],
)

macos_bundle = rule_factory.create_apple_rule(
    doc = "Builds and bundles a macOS Loadable Bundle.",
    implementation = _macos_bundle_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.app_icon_attrs(),
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
            "bundle_extension": attr.string(
                doc = """
The extension, without a leading dot, that will be used to name the bundle. If this attribute is not
set, then the extension will be `.bundle`.
""",
            ),
            "bundle_loader": attr.label(
                doc = """
The target representing the executable that will be loading this bundle. Undefined symbols from the
bundle are checked against this execuable during linking as if it were one of the dynamic libraries
the bundle was linked with.
""",
                providers = [AppleExecutableBinaryInfo],
            ),
        },
    ],
)

macos_extension = rule_factory.create_apple_rule(
    doc = """Builds and bundles a macOS Application Extension.

Most macOS app extensions use a plug-in-based architecture where the
executable's entry point is provided by a system framework. However, macOS 11
introduced Widget Extensions that use a traditional `main` entry
point (typically expressed through Swift's `@main` attribute).""",
    implementation = _macos_extension_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.app_icon_attrs(),
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.extensionkit_attrs(),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.locales_to_include_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, MacosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework))
that this target depends on.
""",
            ),
        },
    ],
)

macos_quick_look_plugin = rule_factory.create_apple_rule(
    doc = "Builds and bundles a macOS Quick Look Plugin.",
    implementation = _macos_quick_look_plugin_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
        },
    ],
)

macos_kernel_extension = rule_factory.create_apple_rule(
    cfg = transition_support.apple_rule_arm64_as_arm64e_transition,
    doc = "Builds and bundles a macOS Kernel Extension.",
    implementation = _macos_kernel_extension_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
        },
    ],
)

macos_spotlight_importer = rule_factory.create_apple_rule(
    doc = "Builds and bundles a macOS Spotlight Importer.",
    implementation = _macos_spotlight_importer_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
        },
    ],
)

macos_xpc_service = rule_factory.create_apple_rule(
    doc = "Builds and bundles a macOS XPC Service.",
    implementation = _macos_xpc_service_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            profile_extension = ".provisionprofile",
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
        },
    ],
)

macos_command_line_application = rule_factory.create_apple_rule(
    doc = """Builds a macOS Command Line Application binary.

A command line application is a standalone binary file, rather than a `.app`
bundle like those produced by [`macos_application`](#macos_application). Unlike
a plain `apple_binary` target, however, this rule supports versioning and
embedding an `Info.plist` into the binary and allows the binary to be
code-signed.

Targets created with `macos_command_line_application` can be executed using
`bazel run`.""",
    implementation = _macos_command_line_application_impl,
    is_executable = True,
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.custom_transition_allowlist_attr(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            supports_capabilities = False,
            profile_extension = ".provisionprofile",
        ),
        {
            "infoplists": attr.label_list(
                allow_files = [".plist"],
                doc = """
A list of .plist files that will be merged to form the Info.plist that represents the application
and is embedded into the binary. Please see
[Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling)
for what is supported.
""",
            ),
            "launchdplists": attr.label_list(
                allow_files = [".plist"],
                doc = """
A list of system wide and per-user daemon/agent configuration files, as specified by the launch
plist manual that can be found via `man launchd.plist`. These are XML files that can be loaded into
launchd with launchctl, and are required of command line applications that are intended to be used
as launch daemons and agents on macOS. All `launchd.plist`s referenced by this attribute will be
merged into a single plist and written directly into the `__TEXT`,`__launchd_plist` section of the
linked binary.
""",
            ),
            "version": attr.label(
                providers = [[AppleBundleVersionInfo]],
                doc = """
An `apple_bundle_version` target that represents the version for this target. See
[`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).
""",
            ),
        },
    ],
)

macos_dylib = rule_factory.create_apple_rule(
    doc = "Builds a macOS Dylib binary.",
    implementation = _macos_dylib_impl,
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.custom_transition_allowlist_attr(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            supports_capabilities = False,
            profile_extension = ".provisionprofile",
        ),
        {
            "infoplists": attr.label_list(
                allow_files = [".plist"],
                doc = """
A list of .plist files that will be merged to form the Info.plist that represents the application
and is embedded into the binary. Please see
[Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling)
for what is supported.
""",
            ),
            "version": attr.label(
                providers = [[AppleBundleVersionInfo]],
                doc = """
An `apple_bundle_version` target that represents the version for this target. See
[`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).
""",
            ),
        },
    ],
)

def _macos_framework_impl(ctx):
    """Experimental implementation of macos_framework."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.framework,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bin_root_path = ctx.bin_dir.path
    bundle_id = ctx.attr.bundle_id
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    executable_name = ctx.attr.executable_name
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    cc_features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        language = "objc",
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    signed_frameworks = []
    if codesigning_support.should_sign_bundles(
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        features = features,
    ):
        signed_frameworks = [bundle_name + bundle_extension]
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = ["resources"],
    )

    extra_linkopts = [
        "-dynamiclib",
        "-Wl,-install_name,@rpath/{name}{extension}/{name}".format(
            extension = bundle_extension,
            name = bundle_name,
        ),
    ]
    if ctx.attr.extension_safe:
        extra_linkopts.append("-fapplication-extension")

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        avoid_deps = ctx.attr.frameworks,
        # Frameworks do not have entitlements.
        entitlements = None,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = extra_linkopts,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive_for_embedding = outputs.archive_for_embedding(
        actions = actions,
        bundle_name = bundle_name,
        bundle_extension = bundle_extension,
        label_name = label.name,
        rule_descriptor = rule_descriptor,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            extension_safe = ctx.attr.extension_safe,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.framework,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            embedded_targets = ctx.attr.frameworks,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        # TODO(kaipi): Check if clang_rt dylibs are needed in Frameworks, or if
        # the can be skipped.
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.frameworks,
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            frameworks = [archive_for_embedding],
            embeddable_targets = ctx.attr.frameworks,
            platform_prerequisites = platform_prerequisites,
            signed_frameworks = depset(signed_frameworks),
        ),
        partials.extension_safe_validation_partial(
            is_extension_safe = ctx.attr.extension_safe,
            rule_label = label,
            targets_to_validate = ctx.attr.frameworks,
        ),
        partials.framework_headers_partial(hdrs = ctx.files.hdrs),
        partials.framework_provider_partial(
            actions = actions,
            bin_root_path = bin_root_path,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            bundle_only = ctx.attr.bundle_only,
            cc_features = cc_features,
            cc_info = link_result.cc_info,
            cc_toolchain = cc_toolchain,
            rule_label = label,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            targets_to_avoid = ctx.attr.frameworks,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
            version_keys_required = False,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            dependency_targets = ctx.attr.frameworks,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.apple_symbols_file_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            dependency_targets = ctx.attr.frameworks,
            dsym_binaries = debug_outputs.dsym_binaries,
            label_name = label.name,
            include_symbols_in_bundle = False,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(files = processor_result.output_files),
        MacosFrameworkBundleInfo(),
        new_appleframeworkbundleinfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ] + processor_result.providers

def _macos_dynamic_framework_impl(ctx):
    """Experimental implementation of macos_dynamic_framework."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.framework,
    )

    # This rule should only have one swift_library dependency. This means len(ctx.attr.deps) should be 1
    swiftdeps = [x for x in ctx.attr.deps if SwiftInfo in x]
    if len(swiftdeps) != 1 or len(ctx.attr.deps) > 1:
        fail(
            """\
                    error: Swift dynamic frameworks expect a single swift_library dependency.
    """,
        )

    binary_target = ctx.attr.deps[0]

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    bin_root_path = ctx.bin_dir.path
    bundle_id = ctx.attr.bundle_id
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    executable_name = ctx.attr.executable_name
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    cc_features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        language = "objc",
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    label = ctx.label
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    predeclared_outputs = ctx.outputs
    provisioning_profile = ctx.file.provisioning_profile
    resource_deps = ctx.attr.deps + ctx.attr.resources
    top_level_infoplists = resources.collect(
        attr = ctx.attr,
        res_attrs = ["infoplists"],
    )
    top_level_resources = resources.collect(
        attr = ctx.attr,
        res_attrs = [
            "app_icons",
            "launch_images",
            "launch_storyboard",
            "strings",
            "resources",
        ],
    )

    signed_frameworks = []
    if codesigning_support.should_sign_bundles(
        provisioning_profile = ctx.file.provisioning_profile,
        rule_descriptor = rule_descriptor,
        features = features,
    ):
        signed_frameworks = [bundle_name + bundle_extension]

    extra_linkopts = [
        "-dynamiclib",
        "-Wl,-install_name,@rpath/{name}{extension}/{name}".format(
            extension = bundle_extension,
            name = bundle_name,
        ),
    ]
    if ctx.attr.extension_safe:
        extra_linkopts.append("-fapplication-extension")

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        avoid_deps = ctx.attr.frameworks,
        # Frameworks do not have entitlements.
        entitlements = None,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_linkopts = extra_linkopts,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    archive_for_embedding = outputs.archive_for_embedding(
        actions = actions,
        bundle_name = bundle_name,
        bundle_extension = bundle_extension,
        label_name = label.name,
        rule_descriptor = rule_descriptor,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
    )

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            executable_name = executable_name,
            extension_safe = ctx.attr.extension_safe,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_location = processor.location.framework,
            bundle_name = bundle_name,
            embed_target_dossiers = False,
            embedded_targets = ctx.attr.frameworks,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
        ),
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.main_thread_checker_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = main_thread_checker_dylibs.get_from_toolchain(ctx),
        ),
        partials.debug_symbols_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            debug_dependencies = ctx.attr.frameworks,
            dsym_binaries = debug_outputs.dsym_binaries,
            dsym_info_plist_template = apple_mac_toolchain_info.dsym_info_plist_template,
            executable_name = executable_name,
            label_name = label.name,
            linkmaps = debug_outputs.linkmaps,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = label,
            version = ctx.attr.version,
        ),
        partials.embedded_bundles_partial(
            frameworks = [archive_for_embedding],
            embeddable_targets = ctx.attr.frameworks,
            platform_prerequisites = platform_prerequisites,
            signed_frameworks = depset(signed_frameworks),
        ),
        partials.extension_safe_validation_partial(
            is_extension_safe = ctx.attr.extension_safe,
            rule_label = label,
            targets_to_validate = ctx.attr.frameworks,
        ),
        partials.framework_provider_partial(
            actions = actions,
            bin_root_path = bin_root_path,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            bundle_only = False,
            cc_features = cc_features,
            cc_info = link_result.cc_info,
            cc_toolchain = cc_toolchain,
            rule_label = label,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            targets_to_avoid = ctx.attr.frameworks,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
            version_keys_required = False,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            dependency_targets = ctx.attr.frameworks,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.swift_dynamic_framework_partial(
            actions = actions,
            bundle_name = bundle_name,
            label_name = label.name,
            swift_dynamic_framework_info = binary_target[SwiftDynamicFrameworkInfo],
        ),
    ]

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        provisioning_profile = provisioning_profile,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    providers = processor_result.providers

    additional_providers = []
    for provider in providers:
        # HACK: this should be updated so we do not need to dynamically check the provider instance.
        # See: https://github.com/bazelbuild/bazel/issues/22095
        if hasattr(provider, "framework_files"):
            # Make the ObjC provider using the framework_files depset found
            # in the AppleDynamicFramework provider. This is to make the
            # macos_dynamic_framework usable as a dependency in swift_library
            libraries_to_link = libraries_to_link_for_dynamic_framework(
                actions = actions,
                cc_toolchain = cc_toolchain,
                feature_configuration = cc_features,
                libraries = provider.framework_files.to_list(),
            )
            additional_providers.append(
                CcInfo(
                    linking_context = cc_common.create_linking_context(
                        linker_inputs = depset([
                            cc_common.create_linker_input(
                                owner = label,
                                libraries = depset(libraries_to_link),
                            ),
                        ]),
                    ),
                ),
            )
    providers.extend(additional_providers)

    return [
        DefaultInfo(files = processor_result.output_files),
        MacosFrameworkBundleInfo(),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
    ] + providers

def _macos_static_framework_impl(ctx):
    """Implementation of macos_static_framework."""
    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = apple_product_type.static_framework,
    )

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    avoid_deps = ctx.attr.avoid_deps
    cc_toolchain_forwarder = ctx.split_attr._cc_toolchain_forwarder
    deps = ctx.attr.deps
    label = ctx.label
    predeclared_outputs = ctx.outputs
    split_deps = ctx.split_attr.deps
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    executable_name = ctx.attr.executable_name
    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
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
        uses_swift = swift_support.uses_swift(ctx.attr.deps),
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )
    resource_deps = ctx.attr.deps + ctx.attr.resources

    archive_result = linking_support.register_static_library_archive_action(
        ctx = ctx,
        cc_toolchains = cc_toolchain_forwarder,
    )
    binary_artifact = archive_result.library

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            product_type = rule_descriptor.product_type,
            rule_descriptor = rule_descriptor,
        ),
        partials.binary_partial(
            actions = actions,
            binary_artifact = binary_artifact,
            bundle_name = bundle_name,
            executable_name = executable_name,
            label_name = label.name,
        ),
    ]

    swift_infos = {}
    if swift_support.uses_swift(deps):
        for split_attr_key, cc_toolchain in cc_toolchain_forwarder.items():
            apple_platform_info = cc_toolchain[ApplePlatformInfo]
            for dep in split_deps[split_attr_key]:
                if SwiftInfo in dep:
                    swift_infos[apple_platform_info.target_arch] = dep[SwiftInfo]

    # If there's any Swift dependencies on the static framework rule, treat it as a Swift static
    # framework.
    if swift_infos:
        processor_partials.append(
            partials.swift_framework_partial(
                actions = actions,
                avoid_deps = avoid_deps,
                bundle_name = bundle_name,
                label_name = label.name,
                swift_infos = swift_infos,
            ),
        )
    else:
        processor_partials.append(
            partials.framework_header_modulemap_partial(
                actions = actions,
                bundle_name = bundle_name,
                hdrs = ctx.files.hdrs,
                label_name = label.name,
                sdk_dylibs = cc_info_support.get_sdk_dylibs(deps = deps),
                sdk_frameworks = cc_info_support.get_sdk_frameworks(deps = deps),
                umbrella_header = ctx.file.umbrella_header,
            ),
        )

    if not ctx.attr.exclude_resources:
        processor_partials.append(partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = None,
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            version = None,
        ))

    processor_result = processor.process(
        actions = actions,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        apple_xplat_toolchain_info = apple_xplat_toolchain_info,
        bundle_extension = bundle_extension,
        bundle_name = bundle_name,
        codesign_inputs = ctx.files.codesign_inputs,
        codesignopts = codesigning_support.codesignopts_from_rule_ctx(ctx),
        features = features,
        ipa_post_processor = ctx.executable.ipa_post_processor,
        partials = processor_partials,
        platform_prerequisites = platform_prerequisites,
        predeclared_outputs = predeclared_outputs,
        process_and_sign_template = apple_mac_toolchain_info.process_and_sign_template,
        rule_descriptor = rule_descriptor,
        rule_label = label,
    )

    return [
        DefaultInfo(files = processor_result.output_files),
        MacosStaticFrameworkBundleInfo(),
        OutputGroupInfo(**processor_result.output_groups),
    ] + processor_result.providers

macos_framework = rule_factory.create_apple_rule(
    doc = """Builds and bundles an macOS Dynamic Framework.

To use this framework for your app and extensions, list it in the `frameworks` attributes
of those `macos_application` and/or `macos_extension` rules.""",
    implementation = _macos_framework_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            supports_capabilities = False,
        ),
        {
            "bundle_only": attr.bool(
                default = False,
                doc = """
Avoid linking the dynamic framework, but still include it in the app. This is useful when you want
to manually dlopen the framework at runtime.
""",
            ),
            "extension_safe": attr.bool(
                default = False,
                doc = """
If true, compiles and links this framework with `-application-extension`, restricting the binary to
use only extension-safe APIs.
""",
            ),
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, MacosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework))
that this target depends on.
""",
            ),
            # TODO(b/250090851): Document this attribute and its limitations.
            "hdrs": attr.label_list(
                allow_files = [".h"],
            ),
        },
    ],
)

macos_dynamic_framework = rule_factory.create_apple_rule(
    doc = "Builds and bundles a macOS dynamic framework that is consumable by Xcode.",
    implementation = _macos_dynamic_framework_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
                swift_dynamic_framework_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
        ),
        rule_attrs.infoplist_attrs(),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            supports_capabilities = False,
        ),
        {
            "bundle_only": attr.bool(
                default = False,
                doc = """
Avoid linking the dynamic framework, but still include it in the app. This is useful when you want
to manually dlopen the framework at runtime.
""",
            ),
            "extension_safe": attr.bool(
                default = False,
                doc = """
If true, compiles and links this framework with `-application-extension`, restricting the binary to
use only extension-safe APIs.
""",
            ),
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, MacosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework))
that this target depends on.
""",
            ),
            # TODO(b/250090851): Document this attribute and its limitations.
            "hdrs": attr.label_list(
                allow_files = [".h"],
            ),
        },
    ],
)

_STATIC_FRAMEWORK_DEPS_CFG = transition_support.apple_platform_split_transition

macos_static_framework = rule_factory.create_apple_rule(
    cfg = transition_support.apple_platforms_rule_base_transition,
    doc = """Builds and bundles a macOS static framework for third-party distribution.

A static framework is bundled like a dynamic framework except that the embedded
binary is a static library rather than a dynamic library. It is intended to
create distributable static SDKs or artifacts that can be easily imported into
other Xcode projects; it is specifically **not** intended to be used as a
dependency of other Bazel targets. For that use case, use the corresponding
`objc_library` targets directly.

Unlike other macOS bundles, the fat binary in an `macos_static_framework` may
simultaneously contain simulator and device architectures (that is, you can
build a single framework artifact that works for all architectures by specifying
`--macos_cpus=x86_64,arm64` when you build).

`macos_static_framework` supports Swift, but there are some constraints:

* `macos_static_framework` with Swift only works with Xcode 12 and above, since
  the required Swift functionality for module compatibility is available in
  Swift 5.1.
* `macos_static_framework` only supports a single direct `swift_library` target
  that does not depend transitively on any other `swift_library` targets. The
  Swift compiler expects a framework to contain a single Swift module, and each
  `swift_library` target is its own module by definition.
* `macos_static_framework` does not support mixed Objective-C and Swift public
  interfaces. This means that the `umbrella_header` and `hdrs` attributes are
  unavailable when using `swift_library` dependencies. You are allowed to depend
  on `objc_library` from the main `swift_library` dependency, but note that only
  the `swift_library`'s public interface will be available to users of the
  static framework.

When using Swift, the `macos_static_framework` bundles `swiftinterface` and
`swiftdocs` file for each of the required architectures. It also bundles an
umbrella header which is the header generated by the single `swift_library`
target. Finally, it also bundles a `module.modulemap` file pointing to the
umbrella header for Objetive-C module compatibility. This umbrella header and
modulemap can be skipped by disabling the `swift.no_generated_header` feature (
i.e. `--features=-swift.no_generated_header`).""",
    implementation = _macos_static_framework_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = _STATIC_FRAMEWORK_DEPS_CFG,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = False,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = _STATIC_FRAMEWORK_DEPS_CFG,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
        ),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        {
            "_emitswiftinterface": attr.bool(
                default = True,
                doc = "Private attribute to generate Swift interfaces for static frameworks.",
            ),
            "avoid_deps": attr.label_list(
                cfg = _STATIC_FRAMEWORK_DEPS_CFG,
                doc = """
A list of library targets on which this framework depends in order to compile, but the transitive
closure of which will not be linked into the framework's binary.
""",
            ),
            "exclude_resources": attr.bool(
                default = False,
                doc = """
Indicates whether resources should be excluded from the bundle. This can be used to avoid
unnecessarily bundling resources if the static framework is being distributed in a different
fashion, such as a Cocoapod.
""",
            ),
            "hdrs": attr.label_list(
                allow_files = [".h"],
                doc = """
A list of `.h` files that will be publicly exposed by this framework. These headers should have
framework-relative imports, and if non-empty, an umbrella header named `%{bundle_name}.h` will also
be generated that imports all of the headers listed here.
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
