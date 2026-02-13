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

"""Rules to generate import-ready XCFrameworks for testing."""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")
load("@build_bazel_rules_swift//swift:swift.bzl", "SwiftInfo")
load(
    "//test/starlark_tests/rules:generation_support.bzl",
    "generation_support",
)

_PLATFORM_TO_SDK = {
    "ios": "iphoneos",
    "ios_simulator": "iphonesimulator",
    "macos": "macosx",
    "tvos": "appletvos",
    "tvos_simulator": "appletvsimulator",
    "watchos": "watchos",
    "watchos_simulator": "watchsimulator",
}

def _platform_to_library_identifier(platform, architectures):
    """Returns an XCFramework library identifier based on archs, environment, and platform.

    Args:
        platform: Target Apple platform for the XCFramework library (e.g. macos, ios)
        architectures: List of architectures supported by the XCFramework library
            (e.g. ['arm64', 'x86_64']).
    Returns:
        A string representing an XCFramework library identifier.
    """
    platform, _, environment = platform.partition("_")
    environment = environment if environment else "device"
    return "{platform}-{architectures}{environment}".format(
        platform = platform,
        architectures = "_".join(sorted(architectures)),
        environment = "-{0}".format(environment) if environment != "device" else "",
    )

def _sdk_for_platform(platform):
    """Returns an Apple platform SDK name for xcrun command flags.

    Args:
        platform: Target Apple platform for the XCFramework library (e.g. macos, ios)
    Returns:
        A string representing an Apple SDK.
    """
    if platform not in _PLATFORM_TO_SDK:
        fail("Invalid platform: %s - must follow format <apple_os>[_<environment>]" % platform)

    return _PLATFORM_TO_SDK[platform]

def _create_xcframework(
        *,
        actions,
        apple_fragment,
        dsyms = {},
        frameworks = {},
        generate_xcframework_xcodebuild_tool,
        headers = [],
        label,
        libraries = [],
        module_interfaces = [],
        target_dir,
        xcode_config):
    """Generates XCFramework using xcodebuild.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        dsyms: A list of dSYM bundles.
        frameworks: Dictionary of framework paths and framework files.
        generate_xcframework_xcodebuild_tool: Tool to run the -create-xcframework command.
        headers: A list of files referencing headers.
        label: Label of the target being built.
        libraries: A list of files referencing static libraries.
        module_interfaces: List of files referencing Swift module interface files.
        target_dir: Path referencing directory of the current target.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.

    Returns:
        List of generated XCFramework files.
    """
    bundle_name = label.name + ".xcframework"
    xcframework_directory = paths.join(target_dir, bundle_name)
    intermediates_directory = paths.join(label.package, "%s-intermediates" % label.name)

    if (frameworks and libraries) or (not frameworks and not libraries):
        fail("Can only generate XCFrameworks using static libraries or dynamic frameworks.")

    info_plist = actions.declare_file(paths.join(bundle_name, "Info.plist"))
    outputs = [info_plist]

    args = actions.args()
    inputs = []

    if libraries:
        inputs.extend(libraries)
        for library in libraries:
            library_relative_path = paths.relativize(library.short_path, intermediates_directory)
            outputs.append(actions.declare_file(library_relative_path, sibling = info_plist))
            args.add("--library", library.path)

            if headers:
                args.add("--headers", paths.join(library.dirname, "Headers"))

    for header in headers:
        inputs.append(header)
        outputs.append(actions.declare_file(
            paths.relativize(header.short_path, intermediates_directory),
            sibling = info_plist,
        ))

    for framework_path, framework_files in frameworks.items():
        inputs.extend(framework_files)
        outputs.extend([
            actions.declare_file(
                paths.relativize(f.short_path, intermediates_directory),
                sibling = info_plist,
            )
            for f in framework_files
        ])
        args.add("--framework", framework_path)

        if dsyms and framework_path in dsyms:
            dsym_files = dsyms[framework_path]

            inputs.extend(dsym_files)

            dsym_bundles_set = sets.make()

            for file in dsym_files:
                dsym_relative_path = paths.relativize(file.short_path, intermediates_directory)
                target_slice, _, remainder = dsym_relative_path.partition("/")
                outputs.append(actions.declare_file(
                    paths.join(target_slice, "dSYMs", remainder),
                    sibling = info_plist,
                ))

                prefix, ext, _ = file.path.partition(".framework.dSYM")
                dsym_bundle_path = prefix + ext

                sets.insert(dsym_bundles_set, dsym_bundle_path)

            dsym_bundles = sets.to_list(dsym_bundles_set)

            for dsym_bundle in dsym_bundles:
                args.add("--debug-symbols", dsym_bundle)

    for module_interface in module_interfaces:
        inputs.append(module_interface)

        # xcodebuild removes swiftmodule files for XCFrameworks.
        # This filters out these files to avoid Bazel errors due
        # no action generating these files.
        if module_interface.extension == "swiftmodule":
            continue
        outputs.append(actions.declare_file(
            paths.relativize(module_interface.short_path, intermediates_directory),
            sibling = info_plist,
        ))

    args.add("--output", xcframework_directory)

    apple_support.run(
        actions = actions,
        apple_fragment = apple_fragment,
        arguments = [args],
        executable = generate_xcframework_xcodebuild_tool,
        execution_requirements = {"no-sandbox": "1"},
        inputs = depset(inputs),
        mnemonic = "GenerateXCFrameworkXcodebuild",
        outputs = outputs,
        progress_message = "Generating XCFramework using xcodebuild",
        xcode_config = xcode_config,
    )

    return outputs

def _generate_dynamic_xcframework_impl(ctx):
    """Implementation of generate_dynamic_xcframework."""
    actions = ctx.actions
    apple_fragment = ctx.fragments.apple
    cpp_fragment = ctx.fragments.cpp
    label = ctx.label
    target_dir = paths.join(ctx.bin_dir.path, label.package)
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    srcs = ctx.files.srcs
    hdrs = ctx.files.hdrs
    platforms = ctx.attr.platforms
    minimum_os_versions = ctx.attr.minimum_os_versions
    include_versioned_frameworks = ctx.attr.include_versioned_frameworks

    if platforms.keys() != minimum_os_versions.keys():
        fail("Attributes: 'platforms' and 'minimum_os_versions' must define the same keys")

    frameworks = {}
    dsyms = {}
    for platform in platforms:
        sdk = _sdk_for_platform(platform)
        architectures = platforms[platform]
        minimum_os_version = minimum_os_versions[platform]
        library_identifier = _platform_to_library_identifier(
            platform = platform,
            architectures = architectures,
        )

        # Compile library
        binary = generation_support.compile_binary(
            actions = actions,
            apple_fragment = apple_fragment,
            archs = architectures,
            hdrs = hdrs,
            label = label,
            minimum_os_version = minimum_os_version,
            sdk = sdk,
            srcs = srcs,
            xcode_config = xcode_config,
        )

        # Create dynamic library
        dynamic_library = generation_support.create_dynamic_library(
            actions = actions,
            apple_fragment = apple_fragment,
            archs = architectures,
            binary = binary,
            label = label,
            minimum_os_version = minimum_os_version,
            sdk = sdk,
            xcode_config = xcode_config,
        )

        # Create (dynamic) framework bundle
        framework_files = generation_support.create_framework(
            actions = actions,
            apple_fragment = apple_fragment,
            base_path = library_identifier,
            bundle_name = label.name,
            headers = hdrs,
            include_versioned_frameworks = include_versioned_frameworks and platform == "macos",
            is_dynamic = True,
            label = label,
            library = dynamic_library,
            target_os = platform,
            xcode_config = xcode_config,
        )

        framework_path = paths.join(
            binary.dirname,
            library_identifier,
            label.name + ".framework",
        )

        frameworks[framework_path] = framework_files

        if cpp_fragment.apple_generate_dsym:
            dsym_files = generation_support.create_dsym(
                actions = ctx.actions,
                apple_fragment = apple_fragment,
                base_path = library_identifier,
                framework_binary = binary,
                label = label,
                xcode_config = xcode_config,
            )
            dsyms[framework_path] = dsym_files

    # Create xcframework bundle
    xcframework_files = _create_xcframework(
        actions = actions,
        apple_fragment = apple_fragment,
        frameworks = frameworks,
        generate_xcframework_xcodebuild_tool = ctx.executable._generate_xcframework_xcodebuild_tool,
        dsyms = dsyms,
        label = label,
        target_dir = target_dir,
        xcode_config = xcode_config,
    )

    return [
        DefaultInfo(
            files = depset(xcframework_files),
        ),
    ]

def _generate_static_xcframework_impl(ctx):
    """Implementation of generate_static_xcframework."""
    actions = ctx.actions
    apple_fragment = ctx.fragments.apple
    label = ctx.label
    target_dir = paths.join(ctx.bin_dir.path, label.package)
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    srcs = ctx.files.srcs
    hdrs = ctx.files.hdrs
    swift_library = ctx.files.swift_library
    include_module_interface_files = ctx.attr.include_module_interface_files

    platforms = ctx.attr.platforms
    minimum_os_versions = ctx.attr.minimum_os_versions
    generate_modulemap = ctx.attr.generate_modulemap

    if not swift_library and platforms.keys() != minimum_os_versions.keys():
        fail("Attributes: 'platforms' and 'minimum_os_versions' must define the same keys")

    if swift_library and len(platforms) > 1:
        fail("Providing a pre-compiled static library is only allowed for a single platform.")

    if swift_library and minimum_os_versions:
        fail("Attributes `minimum_os_versions` and `swift_library` can't be set simulatenously." +
             "\n - Use `minimum_os_versions` when using Objective-C sources for XCFrameworks." +
             "\n - Use `swift_library` when sourcing a previously compiled Swift library.")

    headers = []
    libraries = []
    module_interfaces = []
    modulemaps = []
    umbrella_header = None
    for platform in platforms:
        architectures = platforms[platform]
        library_identifier = _platform_to_library_identifier(
            platform = platform,
            architectures = architectures,
        )

        library_path = library_identifier
        headers_path = paths.join(library_path, "Headers", label.name)

        if not swift_library:
            # Compile library
            minimum_os_version = minimum_os_versions[platform]
            binary = generation_support.compile_binary(
                actions = actions,
                apple_fragment = apple_fragment,
                archs = architectures,
                label = label,
                minimum_os_version = minimum_os_version,
                sdk = _sdk_for_platform(platform),
                srcs = srcs,
                hdrs = hdrs,
                xcode_config = xcode_config,
            )

            # Create static library
            static_library = generation_support.create_static_library(
                actions = actions,
                apple_fragment = apple_fragment,
                binary = binary,
                label = label,
                parent_dir = library_identifier,
                xcode_config = xcode_config,
            )

            # Copy headers and generate umbrella header
            headers.extend([
                generation_support.copy_file(
                    actions = actions,
                    base_path = headers_path,
                    file = header,
                    label = label,
                )
                for header in hdrs
            ])

            umbrella_header = generation_support.generate_umbrella_header(
                actions = actions,
                bundle_name = label.name,
                headers = hdrs,
                headers_path = headers_path,
                label = label,
            )
            headers.append(umbrella_header)
        else:
            # Copy static library to intermediate directory
            static_library = generation_support.copy_file(
                actions = actions,
                base_path = library_path,
                file = generation_support.get_file_with_extension(
                    files = swift_library,
                    extension = "a",
                ),
                label = label,
                target_filename = label.name + ".a",
            )

            # Copy Swift module files to intermediate directory
            if include_module_interface_files:
                swiftmodule_path = paths.join(library_path, label.name + ".swiftmodule")
                module_interfaces = [
                    generation_support.copy_file(
                        actions = actions,
                        base_path = swiftmodule_path,
                        file = interface_file,
                        label = label,
                        target_filename = "{architecture}.{extension}".format(
                            architecture = architectures[0],
                            extension = interface_file.extension,
                        ),
                    )
                    for interface_file in swift_library
                    if interface_file.extension.startswith("swift")
                ]

            # Copy swiftc generated headers to intermediate directory
            headers.append(
                generation_support.copy_file(
                    actions = actions,
                    base_path = headers_path,
                    label = label,
                    file = generation_support.get_file_with_extension(
                        files = swift_library,
                        extension = "h",
                    ),
                    target_filename = label.name + ".h",
                ),
            )

        # Generate Clang modulemap under Headers directory.
        if generate_modulemap:
            modulemaps.append(
                generation_support.generate_module_map(
                    actions = actions,
                    bundle_name = label.name,
                    headers = headers,
                    label = label,
                    module_map_path = headers_path,
                    umbrella_header = umbrella_header,
                ),
            )

        libraries.append(static_library)

    # Create static XCFramework
    xcframework_files = _create_xcframework(
        actions = actions,
        apple_fragment = apple_fragment,
        generate_xcframework_xcodebuild_tool = ctx.executable._generate_xcframework_xcodebuild_tool,
        headers = headers + modulemaps,
        label = label,
        libraries = libraries,
        module_interfaces = module_interfaces,
        target_dir = target_dir,
        xcode_config = xcode_config,
    )

    return [
        DefaultInfo(
            files = depset(xcframework_files),
        ),
    ]

def _generate_static_framework_xcframework_impl(ctx):
    """Implementation of generate_static_framework_xcframework."""
    actions = ctx.actions
    apple_fragment = ctx.fragments.apple
    label = ctx.label
    target_dir = paths.join(ctx.bin_dir.path, label.package)
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    srcs = ctx.files.srcs
    hdrs = ctx.files.hdrs
    platforms = ctx.attr.platforms
    minimum_os_versions = ctx.attr.minimum_os_versions

    if platforms.keys() != minimum_os_versions.keys():
        fail("Attributes: 'platforms' and 'minimum_os_versions' must define the same keys")

    frameworks = {}
    for platform in platforms:
        sdk = _sdk_for_platform(platform)
        architectures = platforms[platform]
        minimum_os_version = minimum_os_versions[platform]
        library_identifier = _platform_to_library_identifier(
            platform = platform,
            architectures = architectures,
        )

        # Compile library
        binary = generation_support.compile_binary(
            actions = actions,
            apple_fragment = apple_fragment,
            archs = architectures,
            hdrs = hdrs,
            label = label,
            minimum_os_version = minimum_os_version,
            sdk = sdk,
            srcs = srcs,
            xcode_config = xcode_config,
        )

        # Create static library
        static_library = generation_support.create_static_library(
            actions = actions,
            apple_fragment = apple_fragment,
            binary = binary,
            label = label,
            xcode_config = xcode_config,
        )

        # Create (static) framework bundle
        framework_files = generation_support.create_framework(
            actions = actions,
            apple_fragment = apple_fragment,
            base_path = library_identifier,
            bundle_name = label.name,
            headers = hdrs,
            label = label,
            is_dynamic = False,
            library = static_library,
            target_os = platform,
            xcode_config = xcode_config,
        )

        framework_path = paths.join(
            binary.dirname,
            library_identifier,
            label.name + ".framework",
        )
        frameworks[framework_path] = framework_files

    # Create xcframework bundle
    xcframework_files = _create_xcframework(
        actions = actions,
        apple_fragment = apple_fragment,
        frameworks = frameworks,
        generate_xcframework_xcodebuild_tool = ctx.executable._generate_xcframework_xcodebuild_tool,
        label = label,
        target_dir = target_dir,
        xcode_config = xcode_config,
    )

    return [
        DefaultInfo(
            files = depset(xcframework_files),
        ),
    ]

_GENERATE_XCFRAMEWORK_TOOL_ATTRS = {
    "_generate_xcframework_xcodebuild_tool": attr.label(
        executable = True,
        cfg = "exec",
        default = Label("//test/starlark_tests/rules:generate_xcframework_xcodebuild_tool"),
    ),
}

generate_dynamic_xcframework = rule(
    doc = "Generates XCFramework with dynamic frameworks using Xcode build utilities.",
    implementation = _generate_dynamic_xcframework_impl,
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        {
            "srcs": attr.label_list(
                doc = "List of source files for compiling Objective-C(++) / Swift binaries.",
                mandatory = True,
                allow_files = True,
            ),
            "hdrs": attr.label_list(
                doc = "Header files for generated XCFrameworks.",
                mandatory = False,
                allow_files = True,
            ),
            "platforms": attr.string_list_dict(
                doc = """
A dictionary of strings indicating which platform variants should be built (with the following
format: <platform>[_<environment>]) as keys, and arrays of strings listing which architectures
should be built for those platform variants as their values.

    platforms = {
        "ios_simulator": [
            "x86_64",
            "arm64",
        ],
        "ios": ["arm64"],
        "watchos_simulator": ["x86_64"],
    },
""",
                mandatory = True,
            ),
            "minimum_os_versions": attr.string_dict(
                doc = """
A dictionary of strings indicating the minimum OS version supported by each platform variant
represented as a dotted version number as values.

    minimum_os_versions = {
        "ios_simulator": "11.0",
        "ios": "11.0",
        "watchos_simulator": "4.0",
    },
""",
                mandatory = True,
            ),
            "generate_modulemap": attr.bool(
                doc = """Flag to indicate if modulemap generation is enabled.""",
                mandatory = False,
                default = True,
            ),
            "include_versioned_frameworks": attr.bool(
                default = True,
                doc = """
Flag to indicate if the framework should include additional versions of the framework under the
Versions directory. This is only supported for macOS platform.
                """,
            ),
        } | _GENERATE_XCFRAMEWORK_TOOL_ATTRS,
    ),
    fragments = ["apple", "cpp"],
)

generate_static_xcframework = rule(
    doc = "Generates XCFramework with a static library using Xcode build utilities.",
    implementation = _generate_static_xcframework_impl,
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        {
            "srcs": attr.label_list(
                doc = "List of source files for compiling Objective-C(++) / Swift binaries.",
                mandatory = False,
                allow_files = True,
            ),
            "hdrs": attr.label_list(
                doc = "Header files for generated XCFrameworks.",
                mandatory = False,
                allow_files = True,
            ),
            "platforms": attr.string_list_dict(
                doc = """
A dictionary of strings indicating which platform variants should be built (with the following
format: <platform>[_<environment>]) as keys, and arrays of strings listing which architectures
should be built for those platform variants as their values.

    platforms = {
        "ios_simulator": [
            "x86_64",
            "arm64",
        ],
        "ios": ["arm64"],
        "watchos_simulator": ["x86_64"],
    },
""",
                mandatory = True,
            ),
            "minimum_os_versions": attr.string_dict(
                doc = """
A dictionary of strings indicating the minimum OS version supported by each platform variant
represented as a dotted version number as values.

    minimum_os_versions = {
        "ios_simulator": "11.0",
        "ios": "11.0",
        "watchos_simulator": "4.0",
    },
""",
                mandatory = False,
            ),
            "generate_modulemap": attr.bool(
                doc = "Flag to indicate if modulemap generation is enabled.",
                mandatory = False,
                default = True,
            ),
            "swift_library": attr.label(
                allow_files = True,
                mandatory = False,
                providers = [SwiftInfo],
                doc = """
Label referencing a `swift_library` target to source static library and module to use for the
generated XCFramework bundle. Target platform and architecture must match with the `platforms`
attribute. This means that if you're building using `bazel build --config=ios_x86_64`, then the
`platforms` attribute must define the following dictionary: {"ios_simulator": ["x86_64"]}.
""",
            ),
            "include_module_interface_files": attr.bool(
                default = True,
                doc = """
Flag to indicate if the Swift module interface files (i.e. `.swiftmodule` directory) from the
`swift_library` target should be included in the XCFramework bundle or discarded for testing
purposes.
""",
            ),
        } | _GENERATE_XCFRAMEWORK_TOOL_ATTRS,
    ),
    fragments = ["apple"],
)

generate_static_framework_xcframework = rule(
    doc = "Generates XCFramework with static frameworks using Xcode build utilities.",
    implementation = _generate_static_framework_xcframework_impl,
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        {
            "srcs": attr.label_list(
                doc = "List of source files for compiling Objective-C(++) / Swift binaries.",
                mandatory = True,
                allow_files = True,
            ),
            "hdrs": attr.label_list(
                doc = "Header files for generated XCFrameworks.",
                mandatory = False,
                allow_files = True,
            ),
            "platforms": attr.string_list_dict(
                doc = """
A dictionary of strings indicating which platform variants should be built (with the following
format: <platform>[_<environment>]) as keys, and arrays of strings listing which architectures
should be built for those platform variants as their values.

    platforms = {
        "ios_simulator": [
            "x86_64",
            "arm64",
        ],
        "ios": ["arm64"],
        "watchos_simulator": ["i386"],
    },
""",
                mandatory = True,
            ),
            "minimum_os_versions": attr.string_dict(
                doc = """
A dictionary of strings indicating the minimum OS version supported by each platform variant
represented as a dotted version number as values.

    minimum_os_versions = {
        "ios_simulator": "11.0",
        "ios": "11.0",
        "watchos_simulator": "4.0",
    },
""",
                mandatory = True,
            ),
        } | _GENERATE_XCFRAMEWORK_TOOL_ATTRS,
    ),
    fragments = ["apple"],
)
