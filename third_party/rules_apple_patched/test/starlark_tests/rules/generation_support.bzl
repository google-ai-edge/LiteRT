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

"""Apple frameworks and XCFramework generation support methods for testing."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")
load(
    "//apple/internal:intermediates.bzl",  # buildifier: disable=bzl-visibility
    "intermediates",
)

_SDK_TO_VERSION_ARG = {
    "iphonesimulator": "-mios-simulator-version-min",
    "iphoneos": "-miphoneos-version-min",
    "macosx": "-mmacos-version-min",
    "appletvsimulator": "-mtvos-simulator-version-min",
    "appletvos": "-mtvos-version-min",
    "watchsimulator": "-mwatchos-simulator-version-min",
    "watchos": "-mwatchos-version-min",
}

_FRAMEWORK_PLIST_TEMPLATE = """
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>{0}</string>
  <key>CFBundleIdentifier</key>
  <string>org.bazel.{0}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>{0}</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
</dict>
</plist>
"""

def _min_version_arg_for_sdk(sdk, minimum_os_version):
    """Returns the clang minimum version argument for a given SDK as a string.

    Args:
        sdk: A string representing an Apple SDK.
        minimum_os_version: Dotted version string for minimum OS version supported by the target.
    Returns:
        A string representing a clang arg for minimum os version of a given Apple SDK.
    """
    return "{0}={1}".format(_SDK_TO_VERSION_ARG[sdk], minimum_os_version)

def _compile_binary(
        *,
        actions,
        apple_fragment,
        archs,
        embed_bitcode = False,
        embed_debug_info = False,
        hdrs,
        label,
        minimum_os_version,
        sdk,
        srcs,
        xcode_config):
    """Compiles binary for a given Apple platform and architectures using Clang.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        archs: List of architectures to compile (e.g. ['arm64', 'x86_64']).
        embed_bitcode: Whether to include bitcode in the binary.
        embed_debug_info: Whether to include debug info in the binary.
        hdrs: List of headers files to compile.
        label: Label of the target being built.
        minimum_os_version: Dotted version string for minimum OS version supported by the target.
        sdk: A string representing an Apple SDK.
        srcs: List of source files to compile.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    Returns:
        A compiled binary file.
    """
    binary = intermediates.file(
        actions = actions,
        file_name = "{sdk}_{archs}_{label}.o".format(
            archs = "_".join(archs),
            label = label.name,
            sdk = sdk,
        ),
        output_discriminator = None,
        target_name = label.name,
    )

    inputs = []
    inputs.extend(srcs)
    inputs.extend(hdrs)

    args = ["/usr/bin/xcrun"]
    args.extend(["-sdk", sdk])
    args.append("clang")
    args.append(_min_version_arg_for_sdk(sdk, minimum_os_version))

    if embed_bitcode:
        args.append("-fembed-bitcode")

    if embed_debug_info:
        args.append("-g")

    for arch in archs:
        args.extend(["-arch", arch])

    for src in srcs:
        args.extend(["-c", src.path])

    args.extend(["-o", binary.path])

    apple_support.run_shell(
        actions = actions,
        apple_fragment = apple_fragment,
        command = " ".join(args),
        inputs = inputs,
        mnemonic = "XcodeToolingClangCompile",
        outputs = [binary],
        progress_message = "Compiling library to Mach-O using clang",
        xcode_config = xcode_config,
        use_default_shell_env = True,
    )

    return binary

def _create_static_library(
        *,
        actions,
        apple_fragment,
        label,
        parent_dir = "",
        binary,
        xcode_config):
    """Creates an Apple static library using libtool.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        binary: A binary file to use for the archive file.
        label: Label of the target being built.
        parent_dir: Optional parent directory name for the generated archive file.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    Returns:
        A static library (archive) file.
    """
    static_library = intermediates.file(
        actions = actions,
        file_name = paths.join(parent_dir, "{}.a".format(label.name)),
        output_discriminator = None,
        target_name = label.name,
    )

    args = ["/usr/bin/xcrun", "libtool", "-static", binary.path, "-o", static_library.path]

    apple_support.run_shell(
        actions = actions,
        apple_fragment = apple_fragment,
        command = " ".join(args),
        inputs = depset([binary]),
        mnemonic = "XcodeToolingLibtool",
        outputs = [static_library],
        progress_message = "Creating static library using libtool",
        xcode_config = xcode_config,
    )

    return static_library

def _create_dynamic_library(
        *,
        actions,
        apple_fragment,
        archs,
        binary,
        label,
        minimum_os_version,
        sdk,
        xcode_config):
    """Creates an Apple dynamic library using Clang for Objective-C(++) sources.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        archs: List of architectures to compile (e.g. ['arm64', 'x86_64']).
        binary: A binary file to use for the archive file.
        label: Label of the target being built.
        minimum_os_version: Dotted version string for minimum OS version supported by the target.
        sdk: A string representing an Apple SDK.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    Returns:
        A dynamic library file.
    """
    dylib_binary = intermediates.file(
        actions = actions,
        file_name = "{sdk}_{archs}_{name}".format(
            archs = "_".join(archs),
            name = label.name,
            sdk = sdk,
        ),
        output_discriminator = None,
        target_name = label.name,
    )

    args = ["/usr/bin/xcrun"]
    args.extend(["-sdk", sdk])
    args.append("clang")
    args.append("-fobjc-link-runtime")
    args.append(_min_version_arg_for_sdk(sdk, minimum_os_version))
    args.append("-dynamiclib")
    args.extend([
        "-install_name",
        "@rpath/{}.framework/{}".format(label.name, label.name),
    ])

    for arch in archs:
        args.extend(["-arch", arch])

    args.append(binary.path)
    args.extend(["-o", dylib_binary.path])

    apple_support.run_shell(
        actions = actions,
        apple_fragment = apple_fragment,
        command = " ".join(args),
        inputs = depset([binary]),
        mnemonic = "XcodeToolingClangDylib",
        outputs = [dylib_binary],
        progress_message = "Creating dynamic library using clang",
        xcode_config = xcode_config,
    )

    return dylib_binary

def _create_framework(
        *,
        actions,
        apple_fragment,
        base_path = "",
        bundle_name,
        label,
        library,
        headers,
        include_resource_bundle = False,
        include_versioned_frameworks = False,
        is_dynamic,
        module_interfaces = [],
        target_os,
        xcode_config):
    """Creates an Apple platform framework bundle.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        base_path: Base path for the generated archive file (optional).
        bundle_name: Name of the framework bundle.
        label: Label of the target being built.
        library: The library for the framework bundle.
        headers: List of header files for the framework bundle.
        include_resource_bundle: Boolean to indicate if a resource bundle should be added to
            the framework bundle (optional).
        include_versioned_frameworks: Boolean to indicate if the framework should include additional
            versions of the framework under the Versions directory.
        is_dynamic: Whether the generated binary is dynamic.
        module_interfaces: List of Swift module interface files for the framework bundle (optional).
        target_os: The target Apple OS for the generated framework bundle.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    Returns:
        List of files for a .framework bundle.
    """
    framework_files = []
    bundle_directory = paths.join(base_path, bundle_name + ".framework")

    is_macos_framework = target_os == "macos"

    # macOS frameworks can include multiple framework versions under:
    #     MyFramework.framework/Versions
    #
    # This directory can contain N framework versions, and special
    # Current version whose contents are symlinks to the effective
    # current framework version files (e.g. Versions/A). Finally,
    # all top-level bundle files, are symlinks to the special
    # Versions/Current directory.
    framework_directories = [bundle_directory]
    versions_directory = paths.join(bundle_directory, "Versions")

    if is_macos_framework and include_versioned_frameworks:
        framework_directories = [
            paths.join(versions_directory, "A"),
            paths.join(versions_directory, "B"),
        ]

    for framework_directory in framework_directories:
        framework_files.append(
            _copy_framework_library(
                actions = actions,
                apple_fragment = apple_fragment,
                bundle_name = bundle_name,
                framework_directory = framework_directory,
                label = label,
                library = library,
                is_dynamic = is_dynamic,
                xcode_config = xcode_config,
            ),
        )

    for framework_directory in framework_directories:
        resources_directory = paths.join(framework_directory, "Resources")
        infoplist_directory = resources_directory if is_macos_framework else framework_directory
        framework_plist = intermediates.file(
            actions = actions,
            file_name = paths.join(infoplist_directory, "Info.plist"),
            output_discriminator = None,
            target_name = label.name,
        )
        actions.write(
            output = framework_plist,
            content = _FRAMEWORK_PLIST_TEMPLATE.format(bundle_name),
        )
        framework_files.append(framework_plist)

    if headers:
        for framework_directory in framework_directories:
            framework_files.extend(
                _copy_framework_headers_and_modulemap(
                    actions = actions,
                    headers = headers,
                    bundle_name = bundle_name,
                    framework_directory = framework_directory,
                    label = label,
                ),
            )

    if module_interfaces:
        for framework_directory in framework_directories:
            modules_path = paths.join(framework_directory, "Modules", bundle_name + ".swiftmodule")
            framework_files.extend([
                _copy_file(
                    actions = actions,
                    base_path = modules_path,
                    file = interface_file,
                    label = label,
                )
                for interface_file in module_interfaces
            ])

    if include_resource_bundle:
        for framework_directory in framework_directories:
            resources_directory = paths.join(framework_directory, "Resources")
            resources_path = paths.join(resources_directory, bundle_name + ".bundle")

            resource_file = intermediates.file(
                actions = actions,
                file_name = paths.join(resources_path, "Info.plist"),
                output_discriminator = None,
                target_name = label.name,
            )
            actions.write(output = resource_file, content = "Mock resource bundle")
            framework_files.append(resource_file)

    if is_macos_framework and include_versioned_frameworks:
        framework_files.extend(
            _create_macos_framework_symlinks(
                actions = actions,
                bundle_directory = bundle_directory,
                framework_files = framework_files,
                label = label,
                versions_directory = versions_directory,
            ),
        )

    return framework_files

def _copy_framework_library(
        *,
        actions,
        apple_fragment,
        bundle_name,
        framework_directory,
        label,
        library,
        is_dynamic,
        xcode_config):
    """Copies a framework library into a target framework directory.

    For macOS frameworks this requires updating the rpath to add the version path.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        bundle_name: Name of the framework/XCFramework bundle.
        framework_directory: Target .framework directory to copy files to.
        label: Label of the target being built.
        library: The library for the framework bundle.
        is_dynamic: Whether the generated binary is dynamic.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    Returns:
        File referencing copied framework library.
    """
    framework_binary = intermediates.file(
        actions = actions,
        file_name = paths.join(framework_directory, bundle_name),
        output_discriminator = None,
        target_name = label.name,
    )

    cp_command = "cp {src} {dest}".format(
        src = library.path,
        dest = framework_binary.path,
    )

    # Copy and modify binary rpath for macOS versioned framework.
    # For all other platforms, copy the framework binary as is.
    if ".framework/Versions/" in framework_directory:
        if is_dynamic:
            install_name_tool_command = "install_name_tool -id {name} {file}".format(
                name = "@rpath/{name}.framework/Versions/{version}/{name}".format(
                    name = bundle_name,
                    version = paths.basename(framework_directory),
                ),
                file = framework_binary.path,
            )
        else:
            install_name_tool_command = "true"
        apple_support.run_shell(
            actions = actions,
            apple_fragment = apple_fragment,
            xcode_config = xcode_config,
            outputs = [framework_binary],
            inputs = [library],
            command = "{cp_command} && {install_name_tool_command}".format(
                cp_command = cp_command,
                install_name_tool_command = install_name_tool_command,
            ),
        )
    else:
        apple_support.run_shell(
            actions = actions,
            apple_fragment = apple_fragment,
            xcode_config = xcode_config,
            outputs = [framework_binary],
            inputs = [library],
            command = cp_command,
        )

    return framework_binary

def _copy_framework_headers_and_modulemap(
        *,
        actions,
        bundle_name,
        framework_directory,
        headers,
        label):
    """Copies headers and generates umbrella header and modulemap for a framework bundle.

    Args:
        actions: The actions provider from `ctx.actions`.
        bundle_name: Name of the framework/XCFramework bundle.
        framework_directory: Target .framework directory to copy files to.
        headers: List of files referencing Objective-C(++) headers for the framework.
        label: Label of the target being built.
    Returns:
        List of files referencing headers and modulemap for the framework bundle.
    """
    headers_path = paths.join(framework_directory, "Headers")
    framework_headers = [
        _copy_file(
            actions = actions,
            base_path = headers_path,
            file = header,
            label = label,
        )
        for header in headers
    ]
    umbrella_header = _generate_umbrella_header(
        actions = actions,
        bundle_name = bundle_name,
        headers = headers,
        headers_path = headers_path,
        label = label,
    )
    framework_headers.append(umbrella_header)

    module_map_path = paths.join(framework_directory, "Modules")
    framework_headers.append(
        _generate_module_map(
            actions = actions,
            bundle_name = bundle_name,
            is_framework_module = True,
            label = label,
            module_map_path = module_map_path,
            umbrella_header = umbrella_header,
        ),
    )

    return framework_headers

def _create_macos_framework_symlinks(
        *,
        actions,
        bundle_directory,
        framework_files,
        label,
        versions_directory):
    """Creates macOS framework symlinks for top-level and Current files.

    Args:
        actions: The actions provider from `ctx.actions`.
        bundle_directory: Top-level directory for the target framework/XCFramework bundle.
        framework_files: Target .framework directory to copy files to.
        label: Label of the target being built.
        versions_directory: 'Versions' directory for the target framework/XCFramework bundle.

    Returns:
        List of files referencing framework symlinks.
    """
    framework_symlinks = []
    version_prefix = ".framework/Versions/A/"

    current_framework_files = [f for f in framework_files if version_prefix in f.short_path]
    current_version_dir = paths.join(versions_directory, "Current")

    # We currently symlink each framework file into Versions/Current/<file_relpath>,
    # instead of only symlinking Versions/Current -> Versions/A, due to Bazel limitations
    # to create symlinks for a given path instead of a file. Once symlinking to a target
    # path is out of experimental, this can be revisited to symlink directories.
    for framework_file in current_framework_files:
        rfind_index = framework_file.short_path.rfind(version_prefix)
        file_relpath = framework_file.short_path[rfind_index + len(version_prefix):]
        file_relpath_dir = paths.dirname(file_relpath)

        versions_current_file = _copy_file(
            actions = actions,
            base_path = paths.join(
                current_version_dir,
                file_relpath_dir,
            ),
            file = framework_file,
            label = label,
        )
        top_level_file = _copy_file(
            actions = actions,
            base_path = paths.join(
                bundle_directory,
                file_relpath_dir,
            ),
            file = versions_current_file,
            label = label,
        )

        framework_symlinks.append(versions_current_file)
        framework_symlinks.append(top_level_file)

    return framework_symlinks

def _dsym_info_plist_content(framework_name):
    return """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleIdentifier</key>
    <string>com.apple.xcode.dsym.{}.framework.dSYM</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>dSYM</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
  </dict>
</plist>
""".format(framework_name)

def _create_dsym(
        *,
        actions,
        apple_fragment,
        base_path = "",
        bundle_name = "",
        framework_binary,
        label,
        xcode_config):
    """
    Generates a dSYM bundle for a framework using dsymutil.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        base_path: Base path for the generated archive file (optional).
        bundle_name: Name of the framework bundle (optional).
        framework_binary: A `File` referring to the framework binary.
        label: Label of the target being built.
        xcode_config: The `apple_common.XcodeVersionConfig` provider from the context.
    """
    bundle_name = bundle_name or label.name
    bundle_directory = paths.join(base_path, bundle_name + ".framework.dSYM")

    # Generate dSYM bundle's DWARF binary
    dsym_binary_file = intermediates.file(
        actions = actions,
        file_name = paths.join(
            bundle_directory,
            "Contents",
            "Resources",
            "DWARF",
            bundle_name,
        ),
        output_discriminator = None,
        target_name = label.name,
    )
    dsym_info_plist = intermediates.file(
        actions = actions,
        file_name = paths.join(
            bundle_directory,
            "Contents",
            "Info.plist",
        ),
        output_discriminator = None,
        target_name = label.name,
    )

    args = actions.args()
    args.add("dsymutil")
    args.add("--flat")
    args.add("--out", dsym_binary_file)
    args.add(framework_binary)

    apple_support.run(
        actions = actions,
        xcode_config = xcode_config,
        apple_fragment = apple_fragment,
        inputs = [framework_binary],
        outputs = [dsym_binary_file],
        executable = "/usr/bin/xcrun",
        arguments = [args],
        mnemonic = "GenerateImportedAppleFrameworkDsym",
    )

    # Write dSYM bundle's Info.plist
    actions.write(
        content = _dsym_info_plist_content(bundle_name),
        output = dsym_info_plist,
    )

    dsym_files = [dsym_binary_file, dsym_info_plist]

    return dsym_files

def _copy_file(*, actions, base_path = "", file, label, target_filename = None):
    """Copies file to a target directory.

    Args:
        actions: The actions provider from `ctx.actions`.
        base_path: Base path for the copied files (optional).
        file: File to copy.
        label: Label of the target being built.
        target_filename: (optional) String for target filename. If None, file basename is used.
    Returns:
        List of copied files.
    """
    filename = target_filename if target_filename else file.basename
    copied_file_path = paths.join(base_path, filename)

    copied_file = intermediates.file(
        actions = actions,
        file_name = copied_file_path,
        output_discriminator = None,
        target_name = label.name,
    )

    actions.run_shell(
        inputs = [file],
        outputs = [copied_file],
        command = "cp {src} {dest}".format(
            src = file.path,
            dest = copied_file.path,
        ),
    )

    return copied_file

def _get_file_with_extension(*, extension, files):
    """Traverse a given file list and return file matching given extension.

    Args:
        extension: File extension to match.
        files: List of files to traverse.
    Returns:
        File matching extension, None otherwise.
    """
    for file in files:
        if file.extension == extension:
            return file
    return None

def _generate_umbrella_header(
        *,
        actions,
        bundle_name,
        headers,
        headers_path,
        label):
    """Generates a single umbrella header given a sequence of header files.

    Args:
        actions: The actions provider from `ctx.actions`.
        bundle_name: Name of the framework/XCFramework bundle.
        headers: List of header files for the framework bundle.
        headers_path: Base path for the generated umbrella header file.
        label: Label of the target being built.
    Returns:
        File for the generated umbrella header.
    """
    header_text = "#import <Foundation/Foundation.h>\n"

    header_prefix = bundle_name
    for header in headers:
        header_text += "#import <{}>\n".format(paths.join(header_prefix, header.basename))

    umbrella_header = intermediates.file(
        actions = actions,
        file_name = paths.join(headers_path, bundle_name + ".h"),
        output_discriminator = None,
        target_name = label.name,
    )
    actions.write(
        output = umbrella_header,
        content = header_text,
    )

    return umbrella_header

def _generate_module_map(
        *,
        actions,
        bundle_name,
        headers = None,
        label,
        is_framework_module = False,
        module_map_path,
        umbrella_header = None):
    """Generates a single module map given a sequence of header files.

    Args:
        actions: The actions provider from `ctx.actions`.
        bundle_name: Name of the framework/XCFramework bundle.
        headers: List of header files to use for the generated modulemap file.
        label: Label of the target being built.
        is_framework_module: Boolean to indicate if the generated modulemap is for a framework.
          Defaults to `False`.
        module_map_path: Base path for the generated modulemap file.
        umbrella_header: Umbrella header file to use for generated modulemap file.
    Returns:
        File for the generated modulemap file.
    """
    modulemap_content = actions.args()
    modulemap_content.set_param_file_format("multiline")

    if is_framework_module:
        modulemap_content.add("framework module %s {" % bundle_name)
    else:
        modulemap_content.add("module %s {" % bundle_name)

    if umbrella_header:
        modulemap_content.add("umbrella header \"%s\"" % umbrella_header.basename)
        modulemap_content.add("export *")
        modulemap_content.add("module * { export * }")
    elif headers:
        for header in headers:
            modulemap_content.add("header \"%s\"" % header.basename)
        modulemap_content.add("requires objc")

    modulemap_content.add("}")

    modulemap_file = intermediates.file(
        actions = actions,
        file_name = paths.join(module_map_path, "module.modulemap"),
        output_discriminator = None,
        target_name = label.name,
    )
    actions.write(output = modulemap_file, content = modulemap_content)

    return modulemap_file

generation_support = struct(
    compile_binary = _compile_binary,
    copy_file = _copy_file,
    create_dsym = _create_dsym,
    create_dynamic_library = _create_dynamic_library,
    create_framework = _create_framework,
    create_static_library = _create_static_library,
    get_file_with_extension = _get_file_with_extension,
    generate_module_map = _generate_module_map,
    generate_umbrella_header = _generate_umbrella_header,
)
