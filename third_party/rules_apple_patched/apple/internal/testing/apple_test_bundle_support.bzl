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

"""Helper methods for implementing the test bundles."""

load(
    "@bazel_skylib//lib:types.bzl",
    "types",
)
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
)
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
    "AppleTestInfo",
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
    "bundling_support",
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
    "//apple/internal:features_support.bzl",
    "features_support",
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
    "new_appleextraoutputsinfo",
    "new_appletestinfo",
)
load(
    "//apple/internal:resources.bzl",
    "resources",
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
    "//apple/internal/utils:clang_rt_dylibs.bzl",
    "clang_rt_dylibs",
)
load(
    "//apple/internal/utils:main_thread_checker_dylibs.bzl",
    "main_thread_checker_dylibs",
)

# Default test bundle ID for tests that don't have a test host or were not given
# a bundle ID.
_DEFAULT_TEST_BUNDLE_ID = "com.bazelbuild.rulesapple.Tests"

def _collect_files(rule_attr, attr_names):
    """Collects files from given attr_names (when present) into a depset."""
    transitive_files = []

    for attr_name in attr_names:
        attr_val = getattr(rule_attr, attr_name, None)
        if not attr_val:
            continue
        attr_val_as_list = attr_val if types.is_list(attr_val) else [attr_val]
        transitive_files.extend([f.files for f in attr_val_as_list])

    return depset(transitive = transitive_files)

def _is_swift_target(target):
    """Returns whether a target directly exports a Swift module."""
    if SwiftInfo not in target:
        return False

    # Containing a SwiftInfo provider is insufficient to determine whether a target exports Swift -
    # need check whether it contains at least one Swift direct module.
    for module in target[SwiftInfo].direct_modules:
        if module.swift != None:
            return True

    return False

def _apple_test_info_aspect_impl(target, ctx):
    """See `test_info_aspect` for full documentation."""
    includes = []
    module_maps = []
    swift_modules = []

    # Not all deps (i.e. source files) will have an AppleTestInfo provider. If the
    # dep doesn't, just filter it out.
    test_infos = [
        x[AppleTestInfo]
        for x in getattr(ctx.rule.attr, "deps", [])
        if AppleTestInfo in x
    ]

    # Collect transitive information from deps.
    for test_info in test_infos:
        includes.append(test_info.includes)
        module_maps.append(test_info.module_maps)
        swift_modules.append(test_info.swift_modules)

    if apple_common.Objc in target:
        objc_provider = target[apple_common.Objc]
        includes.append(objc_provider.strict_include)

    if CcInfo in target:
        cc_info = target[CcInfo]
        includes.append(cc_info.compilation_context.includes)
        includes.append(cc_info.compilation_context.quote_includes)
        includes.append(cc_info.compilation_context.system_includes)

    if _is_swift_target(target):
        all_modules = target[SwiftInfo].transitive_modules.to_list()

        module_swiftmodules = [
            module.swift.swiftmodule
            for module in all_modules
            if module.swift
        ]
        swift_modules.append(depset(module_swiftmodules))

        module_module_maps = [
            module.clang.module_map
            for module in all_modules
            if module.clang and type(module.clang.module_map) == "File"
        ]
        module_maps.append(depset(module_module_maps))

    # Collect sources from the current target. Note that we do not propagate
    # sources transitively as we intentionally only show test sources from the
    # test's first-level of dependencies instead of all transitive dependencies.
    #
    # Group the transitively exported headers into `sources` since we have no
    # need to differentiate between internal headers and transitively exported
    # headers.
    non_arc_sources = _collect_files(ctx.rule.attr, ["non_arc_srcs"])
    sources = _collect_files(ctx.rule.attr, ["srcs", "hdrs", "textual_hdrs"])

    return [new_appletestinfo(
        includes = depset(transitive = includes),
        module_maps = depset(transitive = module_maps),
        non_arc_sources = non_arc_sources,
        sources = sources,
        swift_modules = depset(transitive = swift_modules),
    )]

apple_test_info_aspect = aspect(
    attr_aspects = [
        "deps",
    ],
    doc = """
This aspect walks the dependency graph through the `deps` attribute and collects sources, transitive
includes, transitive module maps, and transitive Swift modules.

This aspect propagates an `AppleTestInfo` provider.
""",
    implementation = _apple_test_info_aspect_impl,
)

def _apple_test_info_provider(deps, test_bundle, test_host):
    """Returns an AppleTestInfo provider by collecting the relevant data from dependencies."""
    dep_labels = []
    swift_infos = []

    transitive_includes = []
    transitive_module_maps = []
    transitive_non_arc_sources = []
    transitive_sources = []
    transitive_swift_modules = []

    for dep in deps:
        dep_labels.append(str(dep.label))

        if SwiftInfo in dep:
            swift_infos.append(dep[SwiftInfo])

        test_info = dep[AppleTestInfo]

        transitive_includes.append(test_info.includes)
        transitive_module_maps.append(test_info.module_maps)
        transitive_non_arc_sources.append(test_info.non_arc_sources)
        transitive_sources.append(test_info.sources)
        transitive_swift_modules.append(test_info.swift_modules)

    # Set module_name only for test targets with a single Swift dependency that
    # contains a single Swift module. This is not used if there are multiple
    # Swift dependencies/modules, as it will not be possible to reduce them into
    # a single Swift module and picking an arbitrary one is fragile.
    module_name = None
    if len(swift_infos) == 1:
        module_names = [x.name for x in swift_infos[0].direct_modules if x.swift]
        if len(module_names) == 1:
            module_name = module_names[0]

    return new_appletestinfo(
        deps = depset(dep_labels),
        includes = depset(transitive = transitive_includes),
        module_maps = depset(transitive = transitive_module_maps),
        module_name = module_name,
        non_arc_sources = depset(transitive = transitive_non_arc_sources),
        sources = depset(transitive = transitive_sources),
        swift_modules = depset(transitive = transitive_swift_modules),
        test_bundle = test_bundle,
        test_host = test_host,
    )

def _computed_test_bundle_id(test_host_bundle_id):
    """Compute a test bundle ID from the test host, or a default if not given."""

    if test_host_bundle_id:
        bundle_id = test_host_bundle_id + "Tests"
    else:
        bundle_id = _DEFAULT_TEST_BUNDLE_ID

    return bundle_id

def _test_host_bundle_id(test_host):
    """Return the bundle ID for the given test host, or None if none was given."""
    if not test_host:
        return None
    test_host_bundle_info = test_host[AppleBundleInfo]
    return test_host_bundle_info.bundle_id

def _apple_test_bundle_impl(*, ctx, product_type):
    """Implementation for bundling XCTest bundles."""
    test_host = ctx.attr.test_host
    test_host_bundle_id = _test_host_bundle_id(test_host)

    rule_descriptor = rule_support.rule_descriptor(
        platform_type = ctx.attr.platform_type,
        product_type = product_type,
    )
    bundle_name, bundle_extension = bundling_support.bundle_full_name(
        custom_bundle_name = ctx.attr.bundle_name,
        label_name = ctx.label.name,
        rule_descriptor = rule_descriptor,
    )
    if ctx.attr.base_bundle_id or ctx.attr.bundle_id:
        bundle_id = bundling_support.bundle_full_id(
            base_bundle_id = ctx.attr.base_bundle_id,
            bundle_id = ctx.attr.bundle_id,
            bundle_id_suffix = ctx.attr.bundle_id_suffix,
            bundle_name = bundle_name,
            suffix_default = ctx.attr._bundle_id_suffix_default,
        )
    else:
        bundle_id = _computed_test_bundle_id(test_host_bundle_id)

    if bundle_id == test_host_bundle_id:
        fail("The test bundle's identifier of '" + bundle_id + "' can't be the " +
             "same as the test host's bundle identifier. Please change one of " +
             "them.")

    actions = ctx.actions
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
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
        # iOS test bundles set `families` as a mandatory attr, other Apple OS test bundles do not
        # have a `families` rule attr.
        device_families = getattr(ctx.attr, "families", rule_descriptor.allowed_device_families),
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

    # For unit tests, only pass the test host as the bundle's loader if it propagates
    # `AppleExecutableBinary`, meaning that it's a binary that *we* built. Test hosts with stub
    # binaries (like a non-single target watchOS app) won't have this. (For UI tests, the test host
    # is never passed as the bundle loader, because the host application is loaded out-of-process.)
    if (
        rule_descriptor.product_type == apple_product_type.unit_test_bundle and
        test_host and AppleExecutableBinaryInfo in test_host and
        ctx.attr.test_host_is_bundle_loader
    ):
        bundle_loader = test_host
    else:
        bundle_loader = None

    extra_linkopts = [
        "-Xlinker",
        "-needed_framework",
        "-Xlinker",
        "XCTest",
        "-framework",
        "XCTest",
        "-bundle",
    ]

    if any([_is_swift_target(dep) for dep in ctx.attr.deps]):
        extra_linkopts.extend([
            "-Xlinker",
            "-needed-lXCTestSwiftSupport",
            "-lXCTestSwiftSupport",
        ])

    extra_link_inputs = []

    if "apple.swizzle_absolute_xcttestsourcelocation" in features:
        # `linking_support.register_binary_linking_action` uses
        # `apple_common.link_multi_arch_binary`, which doesn't allow specifying
        # dependencies (it reads them from `ctx.attr.deps`). So we have to
        # manually link the `_swizzle_absolute_xcttestsourcelocation` library.
        swizzle_lib = ctx.attr._swizzle_absolute_xcttestsourcelocation
        for linker_input in swizzle_lib[CcInfo].linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                static_library = library.static_library
                extra_link_inputs.append(static_library)
                extra_linkopts.append(
                    "-Wl,-force_load,{}".format(static_library.path),
                )
            extra_link_inputs.extend(linker_input.additional_inputs)
            extra_linkopts.extend(linker_input.user_link_flags)

    link_result = linking_support.register_binary_linking_action(
        ctx,
        cc_toolchains = cc_toolchain_forwarder,
        avoid_deps = getattr(ctx.attr, "frameworks", []),
        bundle_loader = bundle_loader,
        # Unit/UI tests do not use entitlements.
        entitlements = None,
        exported_symbols_lists = ctx.files.exported_symbols_lists,
        extra_link_inputs = extra_link_inputs,
        extra_linkopts = extra_linkopts,
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
        stamp = ctx.attr.stamp,
    )
    binary_artifact = link_result.binary
    debug_outputs = linking_support.debug_outputs_by_architecture(link_result.outputs)

    if hasattr(ctx.attr, "additional_contents"):
        debug_dependencies = ctx.attr.additional_contents.keys()
    else:
        debug_dependencies = []
    if test_host:
        debug_dependencies.append(test_host)

    if hasattr(ctx.attr, "frameworks"):
        frameworks = list(ctx.attr.frameworks)
        targets_to_avoid = frameworks
        debug_dependencies.extend(frameworks)
    else:
        targets_to_avoid = []

    if bundle_loader:
        targets_to_avoid.append(bundle_loader)

    embeddable_targets = ctx.attr.deps + getattr(ctx.attr, "frameworks", [])

    processor_partials = [
        partials.apple_bundle_info_partial(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
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
        partials.clang_rt_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            dylibs = clang_rt_dylibs.get_from_toolchain(ctx),
        ),
        partials.codesigning_dossier_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            apple_xplat_toolchain_info = apple_xplat_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            entitlements = None,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
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
            debug_dependencies = debug_dependencies,
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
            embeddable_targets = embeddable_targets,
            platform_prerequisites = platform_prerequisites,
        ),
        partials.framework_import_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            features = features,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
            provisioning_profile = provisioning_profile,
            rule_descriptor = rule_descriptor,
            targets = ctx.attr.deps,
            targets_to_avoid = targets_to_avoid,
        ),
        partials.resources_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            bundle_extension = bundle_extension,
            bundle_id = bundle_id,
            bundle_name = bundle_name,
            environment_plist = ctx.file._environment_plist,
            executable_name = executable_name,
            launch_storyboard = getattr(ctx.file, "launch_storyboard", None),
            platform_prerequisites = platform_prerequisites,
            resource_deps = resource_deps,
            rule_descriptor = rule_descriptor,
            rule_label = label,
            targets_to_avoid = targets_to_avoid,
            top_level_infoplists = top_level_infoplists,
            top_level_resources = top_level_resources,
            version = ctx.attr.version,
            version_keys_required = False,
        ),
        partials.swift_dylibs_partial(
            actions = actions,
            apple_mac_toolchain_info = apple_mac_toolchain_info,
            binary_artifact = binary_artifact,
            bundle_dylibs = True,
            label_name = label.name,
            platform_prerequisites = platform_prerequisites,
        ),
    ]

    if platform_prerequisites.platform_type == apple_common.platform_type.macos:
        processor_partials.append(
            partials.macos_additional_contents_partial(
                additional_contents = ctx.attr.additional_contents,
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

    # The processor outputs has all the extra outputs like dSYM files that we want to propagate, but
    # it also includes the archive artifact. This collects all the files that should be output from
    # the rule (except the archive) so that they're propagated and can be returned by the test
    # target.
    filtered_outputs = [
        x
        for x in processor_result.output_files.to_list()
        if x != archive
    ]

    providers = processor_result.providers
    output_files = processor_result.output_files

    # Symlink the test bundle archive to the output attribute. This is used when having a test such
    # as `ios_unit_test(name = "Foo")` to declare a `:Foo.zip` target.
    actions.symlink(
        target_file = ctx.outputs.archive,
        output = ctx.outputs.test_bundle_output,
    )

    if is_experimental_tree_artifact_enabled(platform_prerequisites = platform_prerequisites):
        test_runner_bundle_output = archive
    else:
        test_runner_bundle_output = ctx.outputs.test_bundle_output

    # Append the AppleTestBundleInfo provider with pointers to the test and host bundles.
    test_host_archive = None
    if test_host:
        test_host_archive = test_host[AppleBundleInfo].archive
    providers.extend([
        _apple_test_info_provider(
            deps = ctx.attr.deps,
            test_bundle = test_runner_bundle_output,
            test_host = test_host_archive,
        ),
        coverage_common.instrumented_files_info(
            ctx,
            dependency_attributes = ["deps", "test_host"],
        ),
        new_appleextraoutputsinfo(files = depset(filtered_outputs)),
        DefaultInfo(
            files = output_files,
            runfiles = ctx.runfiles(
                transitive_files = dsyms,
            ),
        ),
        OutputGroupInfo(
            **outputs.merge_output_groups(
                link_result.output_groups,
                processor_result.output_groups,
            )
        ),
        # TODO(b/228856372): Remove when downstream users are migrated off this provider.
        link_result.debug_outputs_provider,
    ])

    return providers

apple_test_bundle_support = struct(
    apple_test_bundle_impl = _apple_test_bundle_impl,
)
