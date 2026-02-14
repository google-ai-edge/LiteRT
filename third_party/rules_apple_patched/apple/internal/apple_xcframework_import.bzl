# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Implementation of XCFramework import rules."""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_clang_module_aspect",
    "swift_common",
)
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//apple:providers.bzl", "AppleFrameworkImportInfo")
load(
    "//apple/internal:apple_toolchains.bzl",
    "AppleMacToolsToolchainInfo",
    "AppleXPlatToolsToolchainInfo",
)
load(
    "//apple/internal:cc_toolchain_info_support.bzl",
    "cc_toolchain_info_support",
)
load(
    "//apple/internal:experimental.bzl",
    "is_experimental_tree_artifact_enabled",
)
load(
    "//apple/internal:framework_import_support.bzl",
    "framework_import_support",
)
load("//apple/internal:intermediates.bzl", "intermediates")
load(
    "//apple/internal:providers.bzl",
    "AppleDynamicFrameworkInfo",
    "new_appledynamicframeworkinfo",
)
load("//apple/internal:rule_attrs.bzl", "rule_attrs")
load(
    "//apple/internal/aspects:swift_usage_aspect.bzl",
    "SwiftUsageInfo",
)
load(
    "//apple/internal/providers:framework_import_bundle_info.bzl",
    "AppleFrameworkImportBundleInfo",
)

# Currently, XCFramework bundles can contain Apple frameworks or libraries.
# This defines an _enum_ to identify an imported XCFramework bundle type.
_BUNDLE_TYPE = struct(frameworks = 1, libraries = 2)

def _classify_xcframework_imports(config_vars, xcframework_imports):
    """Classifies XCFramework files for later processing.

    Args:
        config_vars: A dict of configuration variables from ctx.var.
        xcframework_imports: List of File for an imported Apple XCFramework.
    Returns:
        A struct containing xcframework import files information:
            - bundle_name: The XCFramework bundle name infered by filepaths.
            - bundle_type: The XCFramework bundle type (frameworks or libraries).
            - files: The XCFramework import files.
            - files_by_category: Classified XCFramework import files.
            - info_plist: The XCFramework bundle Info.plist file.
    """
    info_plist = None
    bundle_name = None

    framework_files = []
    dsym_files = []
    xcframework_files = []
    for file in xcframework_imports:
        parent_dir_name = paths.basename(file.dirname)
        is_bundle_root_file = parent_dir_name.endswith(".xcframework")

        if not info_plist and is_bundle_root_file and file.basename == "Info.plist":
            bundle_name, _ = paths.split_extension(parent_dir_name)
            info_plist = file
            continue

        if ".framework/" in file.short_path:
            framework_files.append(file)
        elif ".framework.dSYM/" in file.short_path:
            dsym_files.append(file)
        else:
            xcframework_files.append(file)

    if not info_plist:
        fail("XCFramework import files doesn't include an Info.plist file")
    if not bundle_name:
        fail("Could not infer XCFramework bundle name from Info.plist file path")

    if framework_files:
        files = framework_files + dsym_files
        bundle_type = _BUNDLE_TYPE.frameworks
        files_by_category = framework_import_support.classify_framework_imports(config_vars, files)
    else:
        files = xcframework_files
        bundle_type = _BUNDLE_TYPE.libraries
        files_by_category = framework_import_support.classify_file_imports(config_vars, files)

    return struct(
        bundle_name = bundle_name,
        bundle_type = bundle_type,
        files = files,
        files_by_category = files_by_category,
        info_plist = info_plist,
    )

def _get_xcframework_library(
        *,
        actions,
        apple_fragment,
        apple_mac_toolchain_info,
        label,
        parse_xcframework_info_plist = False,
        target_triplet,
        xcframework,
        xcode_config):
    """Returns a processed XCFramework library for a given platform.

    Imported XCFramework files are processed through files path parsing, infering the effective
    XCFramework library to use based on target platform, and architecture being built matching
    XCFramework library identifiers.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        apple_mac_toolchain_info: An AppleMacToolsToolchainInfo provider.
        label: Label of the target being built.
        parse_xcframework_info_plist: Boolean to indicate if XCFramework library inferrence should
            be done parsing the XCFramework Info.plist file via the execution-phase tool
            xcframework_processor_tool.py.
        target_triplet: Struct referring a Clang target triplet.
        xcframework: Struct containing imported XCFramework details.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.

    Returns:
        A struct containing processed XCFramework files:
            binary: File referencing the XCFramework library binary.
            framework_imports: List of File referencing XCFramework library files to be bundled
                by a top-level target (ios_application) consuming the target being built.
            framework_includes: List of strings referencing parent directories for framework
                bundles.
            headers: List of File referencing XCFramework library header files. This can be either
                a single tree artifact or a list of regular artifacts.
            clang_module_map: File referencing the XCFramework library Clang modulemap file.
            swift_module_interface: File referencing the XCFramework library Swift module interface
                file (`.swiftinterface`).
    """
    xcframework_library = None
    if not parse_xcframework_info_plist:
        xcframework_library = _get_xcframework_library_from_paths(
            target_triplet = target_triplet,
            xcframework = xcframework,
        )

    if xcframework_library:
        return xcframework_library

    return _get_xcframework_library_with_xcframework_processor(
        actions = actions,
        apple_fragment = apple_fragment,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        label = label,
        target_triplet = target_triplet,
        xcframework = xcframework,
        xcode_config = xcode_config,
    )

def _get_xcframework_library_from_paths(*, target_triplet, xcframework):
    """Infer XCFramework library for the target platform, architecture based on paths.

    Args:
        target_triplet: Struct referring a Clang target triplet.
        xcframework: Struct containing imported XCFramework details.
    Returns:
        A struct containing processed XCFramework files. See _get_xcframework_library.
    """
    library_identifier = _get_library_identifier(
        binary_imports = xcframework.files_by_category.binary_imports,
        bundle_type = xcframework.bundle_type,
        target_architecture = target_triplet.architecture,
        target_environment = target_triplet.environment,
        target_platform = target_triplet.os,
    )

    if not library_identifier:
        return None

    def filter_by_library_identifier(files):
        return [f for f in files if "/{}/".format(library_identifier) in f.short_path]

    files_by_category = xcframework.files_by_category
    framework_binaries = filter_by_library_identifier(files_by_category.binary_imports)
    framework_imports = filter_by_library_identifier(files_by_category.bundling_imports)
    headers = filter_by_library_identifier(files_by_category.header_imports)
    module_maps = filter_by_library_identifier(files_by_category.module_map_imports)
    dsyms = filter_by_library_identifier(files_by_category.dsym_imports)
    dsym_binaries = framework_import_support.get_dsym_binaries(dsyms)
    debug_info_binaries = framework_import_support.get_debug_info_binaries(
        dsym_binaries = dsym_binaries,
        framework_binaries = framework_binaries,
    )

    swiftmodules = framework_import_support.get_swift_module_files_with_target_triplet(
        swift_module_files = filter_by_library_identifier(
            files_by_category.swift_module_imports,
        ),
        target_triplet = target_triplet,
    )

    swift_module_interfaces = framework_import_support.get_swift_module_files_with_target_triplet(
        swift_module_files = filter_by_library_identifier(
            files_by_category.swift_interface_imports,
        ),
        target_triplet = target_triplet,
    )

    framework_files = filter_by_library_identifier(xcframework.files)

    includes = []
    framework_includes = []
    if xcframework.bundle_type == _BUNDLE_TYPE.frameworks:
        framework_includes = [paths.dirname(f.dirname) for f in framework_binaries]
    else:
        # For library XCFrameworks, in Xcode the contents of "Headers" are copied to an intermediate
        # directory for referencing artifacts to include in the build; to replicate this behavior,
        # make sure "includes" is set at the point where "Headers" is found, adjacent to any
        # binaries.
        if headers:
            includes = [paths.join(f.dirname, "Headers") for f in framework_binaries]

    return struct(
        binary = framework_binaries[0],
        debug_info_binaries = debug_info_binaries,
        dsyms = dsyms,
        framework_files = framework_files,
        framework_imports = framework_imports,
        framework_includes = framework_includes,
        headers = headers,
        includes = includes,
        clang_module_map = module_maps[0] if module_maps else None,
        swiftmodule = swiftmodules,
        swift_module_interface = swift_module_interfaces[0] if swift_module_interfaces else None,
    )

def _get_xcframework_library_with_xcframework_processor(
        *,
        actions,
        apple_fragment,
        apple_mac_toolchain_info,
        label,
        target_triplet,
        xcframework,
        xcode_config):
    """Register action to copy the XCFramework library for the target platform, architecture.

    The registered action leverages the xcframework_processor tool to copy the effective XCFramework
    library required for the target being built based on platform, and architecture being targeted.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        apple_mac_toolchain_info: An AppleMacToolsToolchainInfo provider.
        label: Label of the target being built.
        target_triplet: Struct referring a Clang target triplet.
        xcframework: Struct containing imported XCFramework details.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    Returns:
        A struct containing processed XCFramework files. See _get_xcframework_library.
    """
    intermediates_common = {
        "actions": actions,
        "target_name": label.name,
        "output_discriminator": "",
    }

    library_suffix = ".framework" if xcframework.bundle_type == _BUNDLE_TYPE.frameworks else ""
    library_path = xcframework.bundle_name + library_suffix

    framework_imports_dir = intermediates.directory(
        dir_name = paths.join("framework_imports", library_path),
        **intermediates_common
    )

    # The folowing artifacts are declared here to be used later on as inputs for different
    # providers; but not added as arguments to the xcframework_processor tool because it's
    # not really needed if you add the target directory for the copied .framework bundle.
    binary_extension = ".a" if xcframework.bundle_type == _BUNDLE_TYPE.libraries else ""
    binary = intermediates.file(
        file_name = paths.join(library_path, xcframework.bundle_name + binary_extension),
        **intermediates_common
    )
    headers_dir = intermediates.directory(
        dir_name = paths.join(library_path, "Headers"),
        **intermediates_common
    )
    modules_dir_path = paths.join(library_path, "Modules")
    module_map_file = intermediates.file(
        file_name = paths.join(modules_dir_path, "module.modulemap"),
        **intermediates_common
    )

    args = actions.args()
    args.add("--bundle_name", xcframework.bundle_name)
    args.add("--info_plist", xcframework.info_plist.path)

    args.add("--platform", target_triplet.os)
    args.add("--architecture", target_triplet.architecture)
    args.add("--environment", target_triplet.environment)

    files_by_category = xcframework.files_by_category
    args.add_all(files_by_category.binary_imports, before_each = "--binary_file")
    args.add_all(files_by_category.bundling_imports, before_each = "--bundle_file")
    args.add_all(files_by_category.header_imports, before_each = "--header_file")
    args.add_all(files_by_category.module_map_imports, before_each = "--modulemap_file")

    args.add("--binary", binary.path)
    args.add("--library_dir", binary.dirname)
    args.add("--framework_imports_dir", framework_imports_dir.path)

    inputs = []
    inputs.extend(xcframework.files)
    inputs.append(xcframework.info_plist)

    outputs = [
        binary,
        framework_imports_dir,
        headers_dir,
        module_map_file,
    ]

    swiftinterface_file = None
    if files_by_category.swift_interface_imports:
        swiftinterface_path = paths.join(
            modules_dir_path,
            "{module_name}.swiftmodule".format(
                module_name = xcframework.bundle_name,
            ),
            "{architecture}.swiftinterface".format(
                architecture = target_triplet.architecture,
            ),
        )
        swiftinterface_file = intermediates.file(
            file_name = swiftinterface_path,
            **intermediates_common
        )
        args.add_all(
            framework_import_support.get_swift_module_files_with_target_triplet(
                swift_module_files = files_by_category.swift_interface_imports,
                target_triplet = target_triplet,
            ),
            before_each = "--swiftinterface_file",
        )
        outputs.append(swiftinterface_file)

    dsyms = []
    if files_by_category.dsym_imports:
        library_dsym_path = library_path + ".dSYM"
        dsym_imports_dir = intermediates.directory(
            dir_name = paths.join("dsym_imports", library_dsym_path),
            **intermediates_common
        )

        args.add("--dsym_imports_dir", dsym_imports_dir.path)
        args.add_all(files_by_category.dsym_imports, before_each = "--dsym")

        outputs.append(dsym_imports_dir)

        dsyms = [dsym_imports_dir]

    xcframework_processor_tool = apple_mac_toolchain_info.xcframework_processor_tool

    apple_support.run(
        actions = actions,
        apple_fragment = apple_fragment,
        arguments = [args],
        executable = xcframework_processor_tool,
        inputs = inputs,
        mnemonic = "ProcessXCFrameworkFiles",
        outputs = outputs,
        xcode_config = xcode_config,
    )

    includes = []
    framework_includes = []
    if xcframework.bundle_type == _BUNDLE_TYPE.frameworks:
        framework_includes = [paths.dirname(binary.dirname)]
    else:
        includes = [headers_dir.path]

    return struct(
        binary = binary,
        # `debug_info_binaries` cannot be set with the information available to us, given that dSYM paths
        # with the processor tool are opaque during analysis.
        debug_info_binaries = [],
        dsyms = dsyms,
        framework_imports = [framework_imports_dir],
        framework_includes = framework_includes,
        headers = [headers_dir],
        includes = includes,
        clang_module_map = module_map_file,
        swiftmodule = [],
        swift_module_interface = swiftinterface_file,
        framework_files = [],
    )

def _get_library_identifier(
        *,
        binary_imports,
        bundle_type,
        target_architecture,
        target_environment,
        target_platform):
    """Returns an XCFramework library identifier for a given target triplet based on import files.

    Args:
        binary_imports: List of files referencing XCFramework binaries.
        bundle_type: The XCFramework bundle type (frameworks or libraries).
        target_architecture: The target Apple architecture for the target being built (e.g. x86_64,
            arm64).
        target_environment: The target Apple environment for the target being built (e.g. simulator,
            device).
        target_platform: The target Apple platform for the target being built (e.g. macos, ios).
    Returns:
        A string for a XCFramework library identifier.
    """
    if bundle_type == _BUNDLE_TYPE.frameworks:
        library_identifiers = [paths.basename(paths.dirname(f.dirname)) for f in binary_imports]
    elif bundle_type == _BUNDLE_TYPE.libraries:
        library_identifiers = [paths.basename(f.dirname) for f in binary_imports]
    else:
        fail("Unrecognized XCFramework bundle type: %s" % bundle_type)

    for library_identifier in library_identifiers:
        platform, _, suffix = library_identifier.partition("-")
        architectures, _, environment = suffix.partition("-")

        if platform != target_platform:
            continue

        if target_architecture not in architectures:
            continue

        if target_environment == "simulator" and not library_identifier.endswith("-simulator"):
            continue

        # Extra handling of path matching for arm64* architectures.
        if target_architecture == "arm64":
            arm64_index = architectures.find(target_architecture)
            arm64e_index = architectures.find("arm64e")
            arm64_32_index = architectures.find("arm64_32")

            if arm64_index == arm64e_index or arm64_index == arm64_32_index:
                continue

        if target_environment == "device" and not environment:
            return library_identifier

        if target_environment != "device" and target_environment == environment:
            return library_identifier

    return None

def _apple_dynamic_xcframework_import_impl(ctx):
    """Implementation for the apple_dynamic_framework_import rule."""
    actions = ctx.actions
    apple_fragment = ctx.fragments.apple
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    cc_toolchain = find_cpp_toolchain(ctx)
    deps = ctx.attr.deps
    disabled_features = ctx.disabled_features
    features = ctx.features
    label = ctx.label
    xcframework_imports = ctx.files.xcframework_imports
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    # TODO(b/258492867): Add tree artifacts support when Bazel can handle remote actions with
    # symlinks. See https://github.com/bazelbuild/bazel/issues/16361.
    target_triplet = cc_toolchain_info_support.get_apple_clang_triplet(cc_toolchain)
    has_versioned_framework_files = framework_import_support.has_versioned_framework_files(
        xcframework_imports,
    )
    tree_artifact_enabled = (
        apple_xplat_toolchain_info.build_settings.use_tree_artifacts_outputs or
        is_experimental_tree_artifact_enabled(config_vars = ctx.var)
    )
    if target_triplet.os == "macos" and has_versioned_framework_files and tree_artifact_enabled:
        fail("The apple_dynamic_xcframework_import rule does not yet support versioned " +
             "frameworks with the experimental tree artifact feature/build setting. " +
             "Please ensure that the `apple.experimental.tree_artifact_outputs` variable is not " +
             "set to 1 on the command line or in your active build configuration.")

    xcframework = _classify_xcframework_imports(ctx.var, xcframework_imports)
    if xcframework.bundle_type == _BUNDLE_TYPE.libraries:
        fail("Importing XCFrameworks with dynamic libraries is not supported.")

    xcframework_library = _get_xcframework_library(
        actions = actions,
        apple_fragment = apple_fragment,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        label = label,
        parse_xcframework_info_plist = (
            apple_xplat_toolchain_info.build_settings.parse_xcframework_info_plist
        ),
        target_triplet = target_triplet,
        xcframework = xcframework,
        xcode_config = xcode_config,
    )

    providers = []

    # Create AppleFrameworkImportInfo provider
    apple_framework_import_info = framework_import_support.framework_import_info_with_dependencies(
        build_archs = [target_triplet.architecture],
        deps = deps,
        debug_info_binaries = xcframework_library.debug_info_binaries,
        dsyms = xcframework_library.dsyms,
        framework_imports = [xcframework_library.binary] +
                            xcframework_library.framework_imports,
    )
    providers.append(apple_framework_import_info)

    # Create CcInfo provider
    cc_info = framework_import_support.cc_info_with_dependencies(
        actions = actions,
        cc_toolchain = cc_toolchain,
        ctx = ctx,
        deps = deps,
        disabled_features = disabled_features,
        features = features,
        framework_includes = xcframework_library.framework_includes,
        header_imports = xcframework_library.headers,
        kind = "dynamic",
        label = label,
        libraries = [] if ctx.attr.bundle_only else [xcframework_library.binary],
        swiftinterface_imports = [xcframework_library.swift_module_interface] if xcframework_library.swift_module_interface else [],
        swiftmodule_imports = xcframework_library.swiftmodule,
    )
    providers.append(cc_info)

    # Create AppleDynamicFrameworkInfo provider
    apple_dynamic_framework_info = new_appledynamicframeworkinfo(
        cc_info = cc_info,
    )
    providers.append(apple_dynamic_framework_info)

    if "apple._import_framework_via_swiftinterface" in features and xcframework_library.swift_module_interface:
        # Create SwiftInfo provider
        swift_toolchain = swift_common.get_toolchain(ctx)
        providers.append(
            framework_import_support.swift_info_from_module_interface(
                actions = actions,
                ctx = ctx,
                deps = deps,
                disabled_features = disabled_features,
                features = features,
                module_name = xcframework.bundle_name,
                swift_toolchain = swift_toolchain,
                swiftinterface_file = xcframework_library.swift_module_interface,
            ),
        )
    else:
        # Create SwiftInteropInfo provider for swift_clang_module_aspect
        swift_interop_info = framework_import_support.swift_interop_info_with_dependencies(
            deps = deps,
            module_name = xcframework.bundle_name,
            module_map_imports = [xcframework_library.clang_module_map],
        )
        if swift_interop_info:
            providers.append(swift_interop_info)

    return providers

def _apple_static_xcframework_import_impl(ctx):
    """Implementation of apple_static_xcframework_import."""
    actions = ctx.actions
    alwayslink = ctx.attr.alwayslink or getattr(ctx.fragments.objc, "alwayslink_by_default", False)
    apple_fragment = ctx.fragments.apple
    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    cc_toolchain = find_cpp_toolchain(ctx)
    deps = ctx.attr.deps
    disabled_features = ctx.disabled_features
    features = ctx.features
    has_swift = ctx.attr.has_swift
    label = ctx.label
    linkopts = ctx.attr.linkopts
    xcframework_imports = ctx.files.xcframework_imports
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    xcframework = _classify_xcframework_imports(ctx.var, xcframework_imports)
    target_triplet = cc_toolchain_info_support.get_apple_clang_triplet(cc_toolchain)

    xcframework_library = _get_xcframework_library(
        actions = actions,
        apple_fragment = apple_fragment,
        apple_mac_toolchain_info = apple_mac_toolchain_info,
        label = label,
        parse_xcframework_info_plist = (
            apple_xplat_toolchain_info.build_settings.parse_xcframework_info_plist
        ),
        target_triplet = target_triplet,
        xcframework = xcframework,
        xcode_config = xcode_config,
    )

    providers = [
        DefaultInfo(
            files = depset(xcframework_imports),
            runfiles = ctx.runfiles(files = ctx.files.data),
        ),
    ]

    # Create AppleFrameworkImportInfo provider
    apple_framework_import_info = framework_import_support.framework_import_info_with_dependencies(
        build_archs = [apple_fragment.single_arch_cpu],
        deps = deps,
    )
    providers.append(apple_framework_import_info)

    additional_cc_infos = []
    additional_objc_providers = []
    if xcframework.files_by_category.swift_interface_imports or \
       xcframework.files_by_category.swift_module_imports or \
       has_swift:
        swift_toolchain = swift_common.get_toolchain(ctx)
        providers.append(SwiftUsageInfo())

        # The Swift toolchain propagates Swift-specific linker flags (e.g.,
        # library/framework search paths) as an implicit dependency. In the
        # rare case that a binary has a Swift framework import dependency but
        # no other Swift dependencies, make sure we pick those up so that it
        # links to the standard libraries correctly.
        additional_cc_infos.extend(swift_toolchain.implicit_deps_providers.cc_infos)

        # TODO: remove this once rules_swift 3+ is required
        if hasattr(swift_toolchain.implicit_deps_providers, "objc_infos"):
            additional_objc_providers.extend(swift_toolchain.implicit_deps_providers.objc_infos)

    # Create Objc provider
    additional_objc_providers.extend([
        dep[apple_common.Objc]
        for dep in deps
        if apple_common.Objc in dep
    ])

    sdk_linkopts = []
    for dylib in ctx.attr.sdk_dylibs:
        if dylib.startswith("lib"):
            dylib = dylib[3:]
        sdk_linkopts.append("-l%s" % dylib)
    for sdk_framework in ctx.attr.sdk_frameworks:
        sdk_linkopts.append("-framework")
        sdk_linkopts.append(sdk_framework)
    for sdk_framework in ctx.attr.weak_sdk_frameworks:
        sdk_linkopts.append("-weak_framework")
        sdk_linkopts.append(sdk_framework)

    # Create CcInfo provider
    cc_info = framework_import_support.cc_info_with_dependencies(
        actions = actions,
        additional_cc_infos = additional_cc_infos,
        alwayslink = alwayslink,
        cc_toolchain = cc_toolchain,
        ctx = ctx,
        deps = deps,
        disabled_features = disabled_features,
        features = features,
        header_imports = xcframework_library.headers,
        kind = "static",
        label = label,
        libraries = [xcframework_library.binary],
        framework_includes = xcframework_library.framework_includes,
        linkopts = sdk_linkopts + linkopts,
        swiftinterface_imports = [xcframework_library.swift_module_interface] if xcframework_library.swift_module_interface else [],
        swiftmodule_imports = xcframework_library.swiftmodule,
        includes = xcframework_library.includes + ctx.attr.includes,
    )
    providers.append(cc_info)

    if "apple._import_framework_via_swiftinterface" in features and xcframework_library.swift_module_interface:
        # Create SwiftInfo provider
        swift_toolchain = swift_common.get_toolchain(ctx)
        providers.append(
            framework_import_support.swift_info_from_module_interface(
                actions = actions,
                ctx = ctx,
                deps = deps,
                disabled_features = disabled_features,
                features = features,
                module_name = xcframework.bundle_name,
                swift_toolchain = swift_toolchain,
                swiftinterface_file = xcframework_library.swift_module_interface,
            ),
        )
    else:
        # Create SwiftInteropInfo provider for swift_clang_module_aspect
        swift_interop_info = framework_import_support.swift_interop_info_with_dependencies(
            deps = deps,
            module_name = xcframework.bundle_name,
            module_map_imports = [xcframework_library.clang_module_map],
        )
        if swift_interop_info:
            providers.append(swift_interop_info)

    # Create AppleFrameworkImportBundleInfo provider.
    bundle_files = [x for x in xcframework_library.framework_files if ".bundle/" in x.short_path]
    if bundle_files:
        providers.append(AppleFrameworkImportBundleInfo(bundle_files = bundle_files))

    return providers

apple_dynamic_xcframework_import = rule(
    doc = """
This rule encapsulates an already-built XCFramework. Defined by a list of files in a .xcframework
directory. apple_xcframework_import targets need to be added as dependencies to library targets
through the `deps` attribute.

### Example

```bzl
apple_dynamic_xcframework_import(
    name = "my_dynamic_xcframework",
    xcframework_imports = glob(["my_dynamic_framework.xcframework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_dynamic_xcframework",
    ],
)
```
""",
    implementation = _apple_dynamic_xcframework_import_impl,
    attrs = dicts.add(
        rule_attrs.common_tool_attrs(),
        {
            "xcframework_imports": attr.label_list(
                allow_empty = False,
                allow_files = True,
                mandatory = True,
                doc = """
List of files under a .xcframework directory which are provided to Apple based targets that depend
on this target.
""",
            ),
            "deps": attr.label_list(
                doc = """
List of targets that are dependencies of the target being built, which will provide headers and be
linked into that target.
""",
                providers = [
                    [CcInfo],
                    [CcInfo, AppleFrameworkImportInfo],
                ],
                aspects = [swift_clang_module_aspect],
            ),
            "bundle_only": attr.bool(
                default = False,
                doc = """
Avoid linking the dynamic framework, but still include it in the app. This is useful when you want
to manually dlopen the framework at runtime.
""",
            ),
            "library_identifiers": attr.string_dict(
                doc = """
Unnecssary and ignored, will be removed in the future.
""",
            ),
            "_cc_toolchain": attr.label(
                default = "@bazel_tools//tools/cpp:current_cc_toolchain",
                doc = "The C++ toolchain to use.",
            ),
        },
    ),
    fragments = ["apple", "cpp"],
    provides = [
        AppleFrameworkImportInfo,
        CcInfo,
        AppleDynamicFrameworkInfo,
    ],
    toolchains = swift_common.use_toolchain() + use_cpp_toolchain(),
)

apple_static_xcframework_import = rule(
    doc = """
This rule encapsulates an already-built XCFramework with static libraries. Defined by a list of
files in a .xcframework directory. apple_xcframework_import targets need to be added as dependencies
to library targets through the `deps` attribute.
### Examples

```slarlark
apple_static_xcframework_import(
    name = "my_static_xcframework",
    xcframework_imports = glob(["my_static_framework.xcframework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_static_xcframework",
    ],
)
```
""",
    implementation = _apple_static_xcframework_import_impl,
    attrs = dicts.add(
        rule_attrs.common_tool_attrs(),
        {
            "alwayslink": attr.bool(
                default = False,
                doc = """
If true, any binary that depends (directly or indirectly) on this XCFramework will link in all the
object files for the XCFramework bundle, even if some contain no symbols referenced by the binary.
This is useful if your code isn't explicitly called by code in the binary; for example, if you rely
on runtime checks for protocol conformances added in extensions in the library but do not directly
reference any other symbols in the object file that adds that conformance.
""",
            ),
            "deps": attr.label_list(
                doc = """
List of targets that are dependencies of the target being built, which will provide headers and be
linked into that target.
""",
                providers = [
                    [CcInfo],
                    [CcInfo, AppleFrameworkImportInfo],
                ],
                aspects = [swift_clang_module_aspect],
            ),
            "data": attr.label_list(
                allow_files = True,
                doc = """
List of files needed by this target at runtime.

Files and targets named in the `data` attribute will appear in the `*.runfiles`
area of this target, if it has one. This may include data files needed by a
binary or library, or other programs needed by it.
""",
            ),
            "has_swift": attr.bool(
                doc = """
A boolean indicating if the target has Swift source code. This helps flag XCFrameworks that do not
include Swift interface files.
""",
                mandatory = False,
                default = False,
            ),
            "includes": attr.string_list(
                doc = """
List of `#include/#import` search paths to add to this target and all depending
targets.

The paths are interpreted relative to the single platform directory inside the
XCFramework for the platform being built.

These flags are added for this rule and every rule that depends on it. (Note:
not the rules it depends upon!) Be very careful, since this may have
far-reaching effects.
""",
            ),
            "linkopts": attr.string_list(
                mandatory = False,
                doc = """
A list of strings representing extra flags that should be passed to the linker.
""",
            ),
            "sdk_dylibs": attr.string_list(
                doc = """
Names of SDK .dylib libraries to link with. For instance, `libz` or `libarchive`. `libc++` is
included automatically if the binary has any C++ or Objective-C++ sources in its dependency tree.
When linking a binary, all libraries named in that binary's transitive dependency graph are used.
""",
            ),
            "sdk_frameworks": attr.string_list(
                doc = """
Names of SDK frameworks to link with (e.g. `AddressBook`, `QuartzCore`). `UIKit` and `Foundation`
are always included when building for the iOS, tvOS, visionOS and watchOS platforms. For macOS, only
`Foundation` is always included. When linking a top level binary, all SDK frameworks listed in that
binary's transitive dependency graph are linked.
""",
            ),
            "weak_sdk_frameworks": attr.string_list(
                doc = """
Names of SDK frameworks to weakly link with. For instance, `MediaAccessibility`. In difference to
regularly linked SDK frameworks, symbols from weakly linked frameworks do not cause an error if they
are not present at runtime.
""",
            ),
            "xcframework_imports": attr.label_list(
                allow_empty = False,
                allow_files = True,
                mandatory = True,
                doc = """
List of files under a .xcframework directory which are provided to Apple based targets that depend
on this target.
""",
            ),
            "library_identifiers": attr.string_dict(
                doc = """
Unnecssary and ignored, will be removed in the future.
""",
            ),
            "_cc_toolchain": attr.label(
                default = "@bazel_tools//tools/cpp:current_cc_toolchain",
                doc = "The C++ toolchain to use.",
            ),
        },
    ),
    fragments = ["apple", "cpp", "objc"],
    toolchains = swift_common.use_toolchain() + use_cpp_toolchain(),
)
