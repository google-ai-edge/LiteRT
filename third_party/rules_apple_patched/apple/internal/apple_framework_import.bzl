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

"""Implementation of framework import rules."""

load(
    "@bazel_skylib//lib:collections.bzl",
    "collections",
)
load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@bazel_skylib//lib:sets.bzl",
    "sets",
)
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_clang_module_aspect",
    "swift_common",
)
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple:providers.bzl",
    "AppleFrameworkImportInfo",
)
load(
    "//apple:utils.bzl",
    "group_files_by_directory",
)
load(
    "//apple/internal:apple_toolchains.bzl",
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
load(
    "//apple/internal:providers.bzl",
    "new_appledynamicframeworkinfo",
)
load(
    "//apple/internal:rule_attrs.bzl",
    "rule_attrs",
)
load(
    "//apple/internal/aspects:swift_usage_aspect.bzl",
    "SwiftUsageInfo",
)
load(
    "//apple/internal/providers:framework_import_bundle_info.bzl",
    "AppleFrameworkImportBundleInfo",
)

def _swiftmodule_for_cpu(swiftmodule_files, cpu):
    """Select the cpu specific swiftmodule."""

    # The paths will be of the following format:
    #   ABC.framework/Modules/ABC.swiftmodule/<arch>.swiftmodule
    # Where <arch> will be a common arch like x86_64, arm64, etc.
    named_files = {f.basename: f for f in swiftmodule_files}

    module = named_files.get("{}.swiftmodule".format(cpu))
    if not module and cpu == "armv7":
        module = named_files.get("arm.swiftmodule")

    return module

def _all_framework_binaries(frameworks_groups):
    """Returns a list of Files of all imported binaries."""
    binaries = []
    for framework_dir, framework_imports in frameworks_groups.items():
        binary = _get_framework_binary_file(framework_dir, framework_imports.to_list())
        if binary != None:
            binaries.append(binary)

    return binaries

def _get_framework_binary_file(framework_dir, framework_imports):
    """Returns the File that is the framework's binary."""
    framework_name = paths.split_extension(paths.basename(framework_dir))[0]
    framework_path = paths.join(framework_dir, framework_name)
    for framework_import in framework_imports:
        if framework_import.path == framework_path:
            return framework_import

    return None

def _grouped_framework_files(framework_imports):
    """Returns a dictionary of each framework's imports, grouped by path to the .framework root."""
    framework_groups = group_files_by_directory(
        framework_imports,
        ["framework"],
        attr = "framework_imports",
    )

    # Only check for unique basenames of these keys, since it's possible to
    # have targets that glob files from different locations but with the same
    # `.framework` name, causing them to be merged into the same framework
    # during bundling.
    unique_frameworks = collections.uniq(
        [paths.basename(path) for path in framework_groups.keys()],
    )
    if len(unique_frameworks) > 1:
        fail("A framework import target may only include files for a " +
             "single '.framework' bundle.", attr = "framework_imports")

    return framework_groups

def _is_debugging(compilation_mode):
    """Returns `True` if the current compilation mode produces debug info.

    rules_apple specific implementation of rules_swift's `is_debugging`, which
    is not currently exported.

    See: https://github.com/bazelbuild/rules_swift/blob/44146fccd9e56fe1dc650a4e0f21420a503d301c/swift/internal/api.bzl#L315-L326
    """
    return compilation_mode in ("dbg", "fastbuild")

def _ensure_swiftmodule_is_embedded(swiftmodule):
    """Ensures that a `.swiftmodule` file is embedded in a library or binary.

    rules_apple specific implementation of rules_swift's
    `ensure_swiftmodule_is_embedded`, which is not currently exported.

    See: https://github.com/bazelbuild/rules_swift/blob/e78ceb37c401a9bf9e551a6accd1df7d864688d5/swift/internal/debugging.bzl#L20-L47
    """
    return dict(
        linkopt = depset(["-Wl,-add_ast_path,{}".format(swiftmodule.path)]),
        link_inputs = depset([swiftmodule]),
    )

def _framework_search_paths(header_imports):
    """Return the list framework search paths for the headers_imports."""
    if header_imports:
        header_groups = _grouped_framework_files(header_imports)

        search_paths = sets.make()
        for path in header_groups.keys():
            sets.insert(search_paths, paths.dirname(path))
        return sets.to_list(search_paths)
    else:
        return []

def _apple_dynamic_framework_import_impl(ctx):
    """Implementation for the apple_dynamic_framework_import rule."""
    actions = ctx.actions
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    cc_toolchain = find_cpp_toolchain(ctx)
    deps = ctx.attr.deps
    disabled_features = ctx.disabled_features
    features = ctx.features
    framework_imports = ctx.files.framework_imports
    label = ctx.label

    # TODO(b/258492867): Add tree artifacts support when Bazel can handle remote actions with
    # symlinks. See https://github.com/bazelbuild/bazel/issues/16361.
    target_triplet = cc_toolchain_info_support.get_apple_clang_triplet(cc_toolchain)
    has_versioned_framework_files = framework_import_support.has_versioned_framework_files(
        framework_imports,
    )
    tree_artifact_enabled = (
        apple_xplat_toolchain_info.build_settings.use_tree_artifacts_outputs or
        is_experimental_tree_artifact_enabled(config_vars = ctx.var)
    )
    if target_triplet.os == "macos" and has_versioned_framework_files and tree_artifact_enabled:
        fail("The apple_dynamic_framework_import rule does not yet support versioned " +
             "frameworks with the experimental tree artifact feature/build setting. " +
             "Please ensure that the `apple.experimental.tree_artifact_outputs` variable is not " +
             "set to 1 on the command line or in your active build configuration.")

    providers = []
    framework = framework_import_support.classify_framework_imports(
        ctx.var,
        framework_imports,
    )

    dsym_binaries = framework_import_support.get_dsym_binaries(ctx.files.dsym_imports)
    dsym_imports = ctx.files.dsym_imports
    framework_groups = _grouped_framework_files(framework_imports)
    framework_binaries = _all_framework_binaries(framework_groups)
    debug_info_binaries = framework_import_support.get_debug_info_binaries(
        dsym_binaries = dsym_binaries,
        framework_binaries = framework_binaries,
    )

    # Create AppleFrameworkImportInfo provider.
    providers.append(framework_import_support.framework_import_info_with_dependencies(
        build_archs = [target_triplet.architecture],
        deps = deps,
        debug_info_binaries = debug_info_binaries,
        dsyms = dsym_imports,
        framework_imports = (
            framework.binary_imports +
            framework.bundling_imports
        ),
    ))

    # Create CcInfo provider.
    cc_info = framework_import_support.cc_info_with_dependencies(
        actions = actions,
        cc_toolchain = cc_toolchain,
        ctx = ctx,
        deps = deps,
        disabled_features = disabled_features,
        features = features,
        framework_includes = _framework_search_paths(
            framework.header_imports +
            framework.swift_interface_imports +
            framework.swift_module_imports,
        ),
        header_imports = framework.header_imports,
        kind = "dynamic",
        label = label,
        libraries = [] if ctx.attr.bundle_only else framework.binary_imports,
        swiftinterface_imports = framework.swift_interface_imports,
        swiftmodule_imports = framework.swift_module_imports,
    )
    providers.append(cc_info)

    # Create AppleDynamicFramework provider.
    framework_groups = _grouped_framework_files(framework_imports)
    framework_dirs_set = depset(framework_groups.keys())
    providers.append(new_appledynamicframeworkinfo(
        cc_info = cc_info,
        framework_dirs = framework_dirs_set,
        framework_files = depset(framework_imports),
    ))

    if "apple._import_framework_via_swiftinterface" in features and framework.swift_interface_imports:
        # Create SwiftInfo provider
        swift_toolchain = swift_common.get_toolchain(ctx)
        swiftinterface_files = framework_import_support.get_swift_module_files_with_target_triplet(
            swift_module_files = framework.swift_interface_imports,
            target_triplet = target_triplet,
        )
        providers.append(
            framework_import_support.swift_info_from_module_interface(
                actions = actions,
                ctx = ctx,
                deps = deps,
                disabled_features = disabled_features,
                features = features,
                module_name = framework.bundle_name,
                swift_toolchain = swift_toolchain,
                swiftinterface_file = swiftinterface_files[0],
            ),
        )
    else:
        # Create _SwiftInteropInfo provider.
        swift_interop_info = framework_import_support.swift_interop_info_with_dependencies(
            deps = deps,
            module_name = framework.bundle_name,
            module_map_imports = framework.module_map_imports,
        )
        if swift_interop_info:
            providers.append(swift_interop_info)

    return providers

def _apple_static_framework_import_impl(ctx):
    """Implementation for the apple_static_framework_import rule."""
    actions = ctx.actions
    alwayslink = ctx.attr.alwayslink or getattr(ctx.fragments.objc, "alwayslink_by_default", False)
    cc_toolchain = find_cpp_toolchain(ctx)
    compilation_mode = ctx.var["COMPILATION_MODE"]
    deps = ctx.attr.deps
    disabled_features = ctx.disabled_features
    features = ctx.features
    framework_imports = ctx.files.framework_imports
    has_swift = ctx.attr.has_swift
    label = ctx.label
    sdk_dylibs = ctx.attr.sdk_dylibs
    sdk_frameworks = ctx.attr.sdk_frameworks
    weak_sdk_frameworks = ctx.attr.weak_sdk_frameworks

    providers = [
        DefaultInfo(runfiles = ctx.runfiles(files = ctx.files.data)),
    ]

    framework = framework_import_support.classify_framework_imports(
        ctx.var,
        framework_imports,
    )

    # Create AppleFrameworkImportInfo provider
    target_triplet = cc_toolchain_info_support.get_apple_clang_triplet(cc_toolchain)
    providers.append(framework_import_support.framework_import_info_with_dependencies(
        build_archs = [target_triplet.architecture],
        deps = deps,
    ))

    # Collect transitive Objc/CcInfo providers from Swift toolchain
    additional_cc_infos = []
    additional_objc_providers = []
    additional_objc_provider_fields = {}
    if framework.swift_interface_imports or framework.swift_module_imports or has_swift:
        toolchain = swift_common.get_toolchain(ctx)
        providers.append(SwiftUsageInfo())

        # The Swift toolchain propagates Swift-specific linker flags (e.g.,
        # library/framework search paths) as an implicit dependency. In the
        # rare case that a binary has a Swift framework import dependency but
        # no other Swift dependencies, make sure we pick those up so that it
        # links to the standard libraries correctly.
        additional_cc_infos.extend(toolchain.implicit_deps_providers.cc_infos)

        # TODO: remove this once rules_swift 3+ is required
        if hasattr(toolchain.implicit_deps_providers, "objc_infos"):
            additional_objc_providers.extend(toolchain.implicit_deps_providers.objc_infos)

        if _is_debugging(compilation_mode):
            swiftmodule = _swiftmodule_for_cpu(
                framework.swift_module_imports,
                target_triplet.architecture,
            )
            if swiftmodule:
                additional_objc_provider_fields.update(_ensure_swiftmodule_is_embedded(swiftmodule))

    # Create apple_common.Objc provider
    additional_objc_providers.extend([
        dep[apple_common.Objc]
        for dep in deps
        if apple_common.Objc in dep
    ])

    linkopts = []
    if sdk_dylibs:
        for dylib in ctx.attr.sdk_dylibs:
            if dylib.startswith("lib"):
                dylib = dylib[3:]
            linkopts.append("-l%s" % dylib)
    if sdk_frameworks:
        for sdk_framework in ctx.attr.sdk_frameworks:
            linkopts.append("-framework")
            linkopts.append(sdk_framework)
    if weak_sdk_frameworks:
        for sdk_framework in ctx.attr.weak_sdk_frameworks:
            linkopts.append("-weak_framework")
            linkopts.append(sdk_framework)

    # Create CcInfo provider
    providers.append(
        framework_import_support.cc_info_with_dependencies(
            actions = actions,
            additional_cc_infos = additional_cc_infos,
            alwayslink = alwayslink,
            cc_toolchain = cc_toolchain,
            ctx = ctx,
            deps = deps,
            disabled_features = disabled_features,
            features = features,
            framework_includes = _framework_search_paths(
                framework.header_imports +
                framework.swift_interface_imports +
                framework.swift_module_imports,
            ),
            header_imports = framework.header_imports,
            kind = "static",
            label = label,
            libraries = framework.binary_imports,
            linkopts = linkopts,
            swiftinterface_imports = framework.swift_interface_imports,
            swiftmodule_imports = framework.swift_module_imports,
        ),
    )

    if "apple._import_framework_via_swiftinterface" in features and framework.swift_interface_imports:
        # Create SwiftInfo provider
        swift_toolchain = swift_common.get_toolchain(ctx)
        swiftinterface_files = framework_import_support.get_swift_module_files_with_target_triplet(
            swift_module_files = framework.swift_interface_imports,
            target_triplet = target_triplet,
        )
        providers.append(
            framework_import_support.swift_info_from_module_interface(
                actions = actions,
                ctx = ctx,
                deps = deps,
                disabled_features = disabled_features,
                features = features,
                module_name = framework.bundle_name,
                swift_toolchain = swift_toolchain,
                swiftinterface_file = swiftinterface_files[0],
            ),
        )
    else:
        # Create SwiftInteropInfo provider for swift_clang_module_aspect
        swift_interop_info = framework_import_support.swift_interop_info_with_dependencies(
            deps = deps,
            module_name = framework.bundle_name,
            module_map_imports = framework.module_map_imports,
        )
        if swift_interop_info:
            providers.append(swift_interop_info)

    # Create AppleFrameworkImportBundleInfo provider.
    bundle_files = [x for x in framework_imports if ".bundle/" in x.short_path]
    if bundle_files:
        providers.append(AppleFrameworkImportBundleInfo(bundle_files = bundle_files))

    return providers

apple_dynamic_framework_import = rule(
    implementation = _apple_dynamic_framework_import_impl,
    fragments = ["cpp"],
    attrs = dicts.add(
        rule_attrs.common_tool_attrs(),
        {
            "framework_imports": attr.label_list(
                allow_empty = False,
                allow_files = True,
                mandatory = True,
                doc = """
The list of files under a .framework directory which are provided to Apple based targets that depend
on this target.
""",
            ),
            "deps": attr.label_list(
                aspects = [swift_clang_module_aspect],
                doc = """
A list of targets that are dependencies of the target being built, which will be linked into that
target.
""",
                providers = [
                    [CcInfo],
                    [CcInfo, AppleFrameworkImportInfo],
                ],
            ),
            "dsym_imports": attr.label_list(
                allow_files = True,
                doc = """
The list of files under a .dSYM directory, that is the imported framework's dSYM bundle.
""",
            ),
            "bundle_only": attr.bool(
                default = False,
                doc = """
Avoid linking the dynamic framework, but still include it in the app. This is useful when you want
to manually dlopen the framework at runtime.
""",
            ),
            "_cc_toolchain": attr.label(
                default = "@bazel_tools//tools/cpp:current_cc_toolchain",
                doc = "The C++ toolchain to use.",
            ),
        },
    ),
    doc = """
This rule encapsulates an already-built dynamic framework. It is defined by a list of
files in exactly one `.framework` directory. `apple_dynamic_framework_import` targets
need to be added to library targets through the `deps` attribute.
### Examples

```python
apple_dynamic_framework_import(
    name = "my_dynamic_framework",
    framework_imports = glob(["my_dynamic_framework.framework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_dynamic_framework",
    ],
)
```
""",
    toolchains = swift_common.use_toolchain() + use_cpp_toolchain(),
)

apple_static_framework_import = rule(
    implementation = _apple_static_framework_import_impl,
    fragments = ["cpp", "objc"],
    attrs = dicts.add(
        rule_attrs.common_tool_attrs(),
        {
            "framework_imports": attr.label_list(
                allow_empty = False,
                allow_files = True,
                mandatory = True,
                doc = """
The list of files under a .framework directory which are provided to Apple based targets that depend
on this target.
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
are always included when building for the iOS, tvOS, visionOS, and watchOS platforms. For macOS, only
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
            "deps": attr.label_list(
                aspects = [swift_clang_module_aspect],
                doc = """
A list of targets that are dependencies of the target being built, which will provide headers and be
linked into that target.
""",
                providers = [
                    [CcInfo],
                    [CcInfo, AppleFrameworkImportInfo],
                ],
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
            "alwayslink": attr.bool(
                default = False,
                doc = """
If true, any binary that depends (directly or indirectly) on this framework will link in all the
object files for the framework file, even if some contain no symbols referenced by the binary. This
is useful if your code isn't explicitly called by code in the binary; for example, if you rely on
runtime checks for protocol conformances added in extensions in the library but do not directly
reference any other symbols in the object file that adds that conformance.
""",
            ),
            "has_swift": attr.bool(
                doc = """
A boolean indicating if the target has Swift source code. This helps flag Apple frameworks that do
not include Swift interface or Swift module files.
""",
                default = False,
            ),
            "_cc_toolchain": attr.label(
                default = "@bazel_tools//tools/cpp:current_cc_toolchain",
                doc = "The C++ toolchain to use.",
            ),
        },
    ),
    toolchains = swift_common.use_toolchain() + use_cpp_toolchain(),
    doc = """
This rule encapsulates an already-built static framework. It is defined by a list of
files in exactly one `.framework` directory. `apple_static_framework_import` targets
need to be added to library targets through the `deps` attribute.
### Examples

```python
apple_static_framework_import(
    name = "my_static_framework",
    framework_imports = glob(["my_static_framework.framework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_static_framework",
    ],
)
```
""",
)
