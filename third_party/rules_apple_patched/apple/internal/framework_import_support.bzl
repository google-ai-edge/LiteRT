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

"""Support methods for Apple framework import rules."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
    "swift_common",
)
load("@build_bazel_rules_swift//swift:swift_interop_info.bzl", "create_swift_interop_info")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//apple:providers.bzl", "AppleFrameworkImportInfo")
load("//apple:utils.bzl", "group_files_by_directory")
load("//apple/internal:providers.bzl", "new_appleframeworkimportinfo")
load("//apple/internal/utils:bundle_paths.bzl", "bundle_paths")
load("//apple/internal/utils:defines.bzl", "defines")
load("//apple/internal/utils:files.bzl", "files")

def _cc_info_with_dependencies(
        *,
        actions,
        additional_cc_infos = [],
        alwayslink = False,
        cc_toolchain,
        ctx,
        deps,
        disabled_features,
        features,
        framework_includes = [],
        header_imports,
        kind,
        label,
        libraries,
        linkopts = [],
        includes = [],
        swiftinterface_imports = [],
        swiftmodule_imports = [],
        is_framework = True):
    """Returns a new CcInfo which includes transitive Cc dependencies.

    Args:
        actions: The actions provider from `ctx.actions`.
        additional_cc_infos: List of additinal CcInfo providers to use for a merged compilation contexts.
        alwayslink: Boolean to indicate if force_load_library should be set for static frameworks.
        cc_toolchain: CcToolchainInfo provider for current target.
        ctx: The Starlark context for a rule target being built.
        deps: List of dependencies for a given target to retrieve transitive CcInfo providers.
        disabled_features: List of features to be disabled for cc_common.compile
        features: List of features to be enabled for cc_common.compile.
        framework_includes: List of Apple framework search paths (defaults to: []).
        header_imports: List of imported header files.
        includes: List of included headers search paths (defaults to: []).
        kind: whether the framework is "static" or "dynamic".
        label: Label of the target being built.
        libraries: The list of framework libraries.
        linkopts: List of linker flags strings to propagate as linker input.
        swiftinterface_imports: List of imported Swift interface files to include
            during build phase, but aren't processed in any way.
        swiftmodule_imports: List of imported Swift module files to include during build phase,
            but aren't processed in any way.
        is_framework: Whether the target is a framework vs library.
    Returns:
        CcInfo provider.
    """
    all_cc_infos = [dep[CcInfo] for dep in deps] + additional_cc_infos
    dep_compilation_contexts = [cc_info.compilation_context for cc_info in all_cc_infos]

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        language = "objc",
        requested_features = features,
        unsupported_features = disabled_features,
    )

    public_hdrs = []
    public_hdrs.extend(header_imports)
    public_hdrs.extend(swiftmodule_imports)
    public_hdrs.extend(swiftinterface_imports)
    (compilation_context, _compilation_outputs) = cc_common.compile(
        name = label.name,
        actions = actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        public_hdrs = public_hdrs,
        framework_includes = framework_includes if is_framework else [],
        includes = includes,
        compilation_contexts = dep_compilation_contexts,
        language = "objc",
    )

    linking_contexts = [cc_info.linking_context for cc_info in all_cc_infos]

    if kind == "static":
        libraries_to_link = _libraries_to_link_for_static_framework(
            actions = actions,
            alwayslink = alwayslink,
            libraries = libraries,
        )
    else:
        libraries_to_link = libraries_to_link_for_dynamic_framework(
            actions = actions,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            libraries = libraries,
        )
    linking_contexts.append(
        cc_common.create_linking_context(
            linker_inputs = depset([
                cc_common.create_linker_input(
                    owner = label,
                    libraries = depset(libraries_to_link),
                    user_link_flags = linkopts,
                ),
            ]),
        ),
    )

    linking_context = cc_common.merge_linking_contexts(
        linking_contexts = linking_contexts,
    )

    return CcInfo(
        compilation_context = compilation_context,
        linking_context = linking_context,
    )

def _classify_file_imports(config_vars, import_files):
    """Classifies a list of imported files based on extension, and paths.

    This support method is used to classify import files for Apple frameworks and XCFrameworks.
    Any file that does not match any known extension will be added to an bundling_imports bucket.

    Args:
        config_vars: A dictionary of configuration variables from ctx.var.
        import_files: List of File to classify.
    Returns:
        A struct containing classified import files by categories:
            - header_imports: Objective-C(++) header imports.
            - module_map_imports: Clang modulemap imports.
            - swift_module_imports: Swift module imports.
            - swift_interface_imports: Swift module interface imports.
            - dsym_imports: dSYM imports.
            - bundling_imports: Unclassified imports.
    """
    bundling_imports = []
    binary_imports = []
    dsym_imports = []
    header_imports = []
    module_map_imports = []
    swift_module_imports = []
    swift_interface_imports = []
    for file in import_files:
        # Extension matching
        file_extension = file.extension
        if file_extension == "h":
            header_imports.append(file)
            continue
        if file_extension == "modulemap":
            # With the flip of `--incompatible_objc_framework_cleanup`, the
            # `objc_library` implementation in Bazel no longer passes module
            # maps as inputs to the compile actions, so that `@import`
            # statements for user-provided framework no longer work in a
            # sandbox. This trap door allows users to continue using `@import`
            # statements for imported framework by adding module map to
            # header_imports so that they are included in Obj-C compilation but
            # they aren't processed in any way.
            if defines.bool_value(
                config_vars = config_vars,
                define_name = "apple.incompatible.objc_framework_propagate_modulemap",
                default = False,
            ):
                header_imports.append(file)
            module_map_imports.append(file)
            continue
        if file_extension == "swiftmodule":
            swift_module_imports.append(file)
            continue
        if file_extension == "swiftinterface" and ".framework.dSYM/" not in file.short_path:
            swift_interface_imports.append(file)
            continue
        if file_extension in ["swiftdoc", "swiftsourceinfo"]:
            # Ignore swiftdoc files, they don't matter in the build, only for IDEs.
            continue
        if file_extension == "a":
            binary_imports.append(file)
            continue
        if ".framework.dSYM/" in file.short_path:
            dsym_imports.append(file)
            continue

        # Path matching
        framework_relative_path = file.short_path.split(".framework/")[-1]
        if framework_relative_path.startswith("Headers/"):
            header_imports.append(file)
            continue
        if framework_relative_path.startswith("Modules/") and framework_relative_path.endswith(".abi.json"):
            # Ignore abi.json files, as they don't matter in the build, and are most commonly used to detect source-breaking API changes during the evolution of a Swift library.
            # See: https://github.com/swiftlang/swift/blob/main/lib/DriverTool/swift_api_digester_main.cpp
            continue

        # Unknown file type, sending to unknown (i.e. resources, Info.plist, etc.)
        bundling_imports.append(file)

    return struct(
        binary_imports = binary_imports,
        dsym_imports = dsym_imports,
        header_imports = header_imports,
        module_map_imports = module_map_imports,
        swift_interface_imports = swift_interface_imports,
        swift_module_imports = swift_module_imports,
        bundling_imports = bundling_imports,
    )

def _classify_framework_imports(config_vars, framework_imports):
    """Classify a list of files referencing an Apple framework.

    Args:
        config_vars: A dictionary (String to String) of configuration variables. Can be from ctx.var.
        framework_imports: List of File for an imported Apple framework.
    Returns:
        A struct containing classified framework import files by categories:
            - bundle_name: The framework bundle name infered by filepaths.
            - binary_imports: Apple framework binary imports.
            - bundling_imports: Apple framework bundle imports.
            - header_imports: Apple framework header imports.
            - module_map_imports: Apple framework modulemap imports.
            - swift_module_imports: Apple framework swiftmodule imports.
            - swift_interface_imports: Apple framework Swift module interface imports.
    """
    framework_imports_by_category = _classify_file_imports(config_vars, framework_imports)

    bundle_name = None
    bundling_imports = []
    binary_imports = []
    for file in framework_imports_by_category.bundling_imports:
        # Infer framework bundle name and binary
        parent_dir_name = paths.basename(file.dirname)
        is_bundle_root_file = parent_dir_name.endswith(".framework")
        if is_bundle_root_file:
            bundle_name, _ = paths.split_extension(parent_dir_name)
            if file.basename == bundle_name:
                binary_imports.append(file)
                continue

        bundling_imports.append(file)

    # TODO: Enable these checks once static library support works with them
    # if not bundle_name:
    #     fail("Could not infer Apple framework name from unclassified framework import files.")
    # if not binary_imports:
    #     fail("Could not find Apple framework binary from framework import files.")

    return struct(
        bundle_name = bundle_name,
        binary_imports = binary_imports,
        bundling_imports = bundling_imports,
        dsym_imports = framework_imports_by_category.dsym_imports,
        header_imports = framework_imports_by_category.header_imports,
        module_map_imports = framework_imports_by_category.module_map_imports,
        swift_interface_imports = framework_imports_by_category.swift_interface_imports,
        swift_module_imports = framework_imports_by_category.swift_module_imports,
    )

def libraries_to_link_for_dynamic_framework(
        *,
        actions,
        cc_toolchain,
        feature_configuration,
        libraries):
    """Return a list of library_to_link's for a dynamic framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        cc_toolchain: CcToolchainInfo provider for current target.
        feature_configuration: The cc enabled features.
        libraries: List of dynamic libraries.

    Returns:
        A list of library_to_link's.
    """
    libraries_to_link = []
    for library in libraries:
        library_to_link = cc_common.create_library_to_link(
            actions = actions,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            dynamic_library = library,
        )
        libraries_to_link.append(library_to_link)

    return libraries_to_link

def _libraries_to_link_for_static_framework(
        *,
        actions,
        alwayslink,
        libraries):
    """Return a list of library_to_link's for a static framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        alwayslink: Whather the libraries should be always linked.
        libraries: List of static libraries.

    Returns:
        A list of library_to_link's.
    """
    libraries_to_link = []
    for library in libraries:
        library_to_link = cc_common.create_library_to_link(
            actions = actions,
            alwayslink = alwayslink,
            static_library = library,
        )
        libraries_to_link.append(library_to_link)

    return libraries_to_link

def _framework_import_info_with_dependencies(
        *,
        build_archs,
        deps,
        debug_info_binaries = [],
        dsyms = [],
        framework_imports = []):
    """Returns AppleFrameworkImportInfo containing transitive framework imports and build archs.

    Args:
        build_archs: List of supported architectures for the imported framework.
        deps: List of transitive dependencies of the current target.
        debug_info_binaries: List of debug info binaries for the imported Framework.
        dsyms: List of dSYM files for the imported Framework.
        framework_imports: List of files to bundle for the imported framework.
    Returns:
        AppleFrameworkImportInfo provider.
    """
    transitive_framework_imports = [
        dep[AppleFrameworkImportInfo].framework_imports
        for dep in deps
        if (AppleFrameworkImportInfo in dep and
            hasattr(dep[AppleFrameworkImportInfo], "framework_imports"))
    ]

    return new_appleframeworkimportinfo(
        build_archs = depset(build_archs),
        debug_info_binaries = depset(debug_info_binaries),
        dsym_imports = depset(dsyms),
        framework_imports = depset(
            framework_imports,
            transitive = transitive_framework_imports,
        ),
    )

def _get_swift_module_files_with_target_triplet(target_triplet, swift_module_files):
    """Filters Swift module files for a target triplet.

    Traverses a list of Swift module files (.swiftdoc, .swiftinterface, .swiftmodule) and selects
    the effective files based on target triplet. This method supports filtering for multiple
    Swift module directories (e.g. XCFramework bundles).

    Args:
        target_triplet: Effective target triplet from CcToolchainInfo provider.
        swift_module_files: List of Swift module files to filter using target triplet.
    Returns:
        List of Swift module files for given target_triplet.
    """
    files_by_module = group_files_by_directory(
        files = swift_module_files,
        extensions = ["swiftmodule"],
        attr = "swift_module_files",
    )

    filtered_files = []
    for _module, module_files in files_by_module.items():
        # Environment suffix is stripped for device interfaces.
        environment = ""
        if target_triplet.environment != "device":
            environment = "-" + target_triplet.environment

        target_triplet_file = files.get_file_with_name(
            files = module_files.to_list(),
            name = "{architecture}-{vendor}-{os}{environment}".format(
                architecture = target_triplet.architecture,
                environment = environment,
                os = target_triplet.os,
                vendor = target_triplet.vendor,
            ),
        )
        architecture_file = files.get_file_with_name(
            files = module_files.to_list(),
            name = target_triplet.architecture,
        )
        filtered_files.append(target_triplet_file or architecture_file)

    return filtered_files

def _get_dsym_binaries(dsym_imports):
    """Returns a list of Files of all imported dSYM binaries."""
    return [
        file
        for file in dsym_imports
        if file.basename.lower() != "info.plist"
    ]

def _get_debug_info_binaries(dsym_binaries, framework_binaries):
    """Return the list of files that provide debug info."""
    all_binaries_dict = {}

    for file in dsym_binaries:
        dsym_bundle_path = bundle_paths.farthest_parent(
            file.short_path,
            "framework.dSYM",
        )
        dsym_bundle_basename = paths.basename(dsym_bundle_path)
        framework_basename = dsym_bundle_basename.rstrip(".dSYM")
        if framework_basename not in all_binaries_dict:
            all_binaries_dict[framework_basename] = file

    for file in framework_binaries:
        if ".framework/" not in file.short_path:
            continue
        framework_path = bundle_paths.farthest_parent(
            file.short_path,
            "framework",
        )
        framework_basename = paths.basename(framework_path)
        if framework_basename not in all_binaries_dict:
            all_binaries_dict[framework_basename] = file

    return all_binaries_dict.values()

def _has_versioned_framework_files(framework_files):
    """Returns True if there are any versioned framework files (i.e. under Versions/ directory).

    Args:
        framework_files: List of File references for imported framework or XCFramework files.
    Returns:
        True if framework files include any versioned frameworks. False otherwise.
    """
    for f in framework_files:
        if ".framework/Versions/" in f.short_path:
            return True
    return False

def _swift_info_from_module_interface(
        *,
        actions,
        ctx,
        deps,
        disabled_features,
        features,
        module_name,
        swift_toolchain,
        swiftinterface_file):
    """Returns SwiftInfo provider for a pre-compiled Swift module compiling it's interface file.


    Args:
        actions: The actions provider from `ctx.actions`.
        ctx: The Starlark context for a rule target being built.
        deps: List of dependencies for a given target to retrieve transitive CcInfo providers.
        disabled_features: List of features to be disabled for cc_common.compile
        features: List of features to be enabled for cc_common.compile.
        module_name: Swift module name.
        swift_toolchain: SwiftToolchainInfo provider for current target.
        swiftinterface_file: `.swiftinterface` File to compile.
    Returns:
        A SwiftInfo provider.
    """
    swift_infos = [dep[SwiftInfo] for dep in deps if SwiftInfo in dep]
    module_context = swift_common.compile_module_interface(
        actions = actions,
        compilation_contexts = [
            dep[CcInfo].compilation_context
            for dep in deps
            if CcInfo in dep
        ],
        feature_configuration = swift_common.configure_features(
            ctx = ctx,
            swift_toolchain = swift_toolchain,
            requested_features = features,
            unsupported_features = disabled_features,
        ),
        module_name = module_name,
        swiftinterface_file = swiftinterface_file,
        swift_infos = swift_infos,
        swift_toolchain = swift_toolchain,
        target_name = ctx.label.name,
    )

    # TODO: use SwiftInfo directly when rules_apple sets min for rules_swift to v3+
    if hasattr(swift_common, "create_swift_info"):
        return swift_common.create_swift_info(
            modules = [module_context],
            swift_infos = swift_infos,
        )
    else:
        return SwiftInfo(
            modules = [module_context],
            swift_infos = swift_infos,
        )

def _swift_interop_info_with_dependencies(deps, module_name, module_map_imports):
    """Return a Swift interop provider for the framework if it has a module map."""
    if not module_map_imports:
        return None

    # Assume that there is only a single module map file (the legacy
    # implementation that read from the Objc provider made the same
    # assumption).

    # TODO: Use the free function when rules_apple sets min for rules_swift to v3+
    if hasattr(swift_common, "create_swift_interop_info"):
        return swift_common.create_swift_interop_info(
            module_map = module_map_imports[0],
            module_name = module_name,
            swift_infos = [dep[SwiftInfo] for dep in deps if SwiftInfo in dep],
        )
    else:
        return create_swift_interop_info(
            module_map = module_map_imports[0],
            module_name = module_name,
            swift_infos = [dep[SwiftInfo] for dep in deps if SwiftInfo in dep],
        )

framework_import_support = struct(
    cc_info_with_dependencies = _cc_info_with_dependencies,
    classify_file_imports = _classify_file_imports,
    classify_framework_imports = _classify_framework_imports,
    framework_import_info_with_dependencies = _framework_import_info_with_dependencies,
    get_swift_module_files_with_target_triplet = _get_swift_module_files_with_target_triplet,
    get_dsym_binaries = _get_dsym_binaries,
    get_debug_info_binaries = _get_debug_info_binaries,
    has_versioned_framework_files = _has_versioned_framework_files,
    swift_info_from_module_interface = _swift_info_from_module_interface,
    swift_interop_info_with_dependencies = _swift_interop_info_with_dependencies,
)
