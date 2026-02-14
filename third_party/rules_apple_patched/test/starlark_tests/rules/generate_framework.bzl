# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Rules to generate import-ready frameworks for testing."""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")
load("@build_bazel_rules_swift//swift:swift.bzl", "SwiftInfo")
load(
    "//test/starlark_tests/rules:generation_support.bzl",
    "generation_support",
)

_SDK_TO_OS = {
    "iphonesimulator": "ios",
    "iphoneos": "ios",
    "macosx": "macos",
    "appletvsimulator": "tvos",
    "appletvos": "tvos",
    "watchsimulator": "watchos",
    "watchos": "watchos",
}

def _generate_import_framework_impl(ctx):
    actions = ctx.actions
    apple_fragment = ctx.fragments.apple
    label = ctx.label
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    architectures = ctx.attr.archs
    hdrs = ctx.files.hdrs
    libtype = ctx.attr.libtype
    minimum_os_version = ctx.attr.minimum_os_version
    sdk = ctx.attr.sdk
    srcs = ctx.files.src

    swift_library_files = ctx.files.swift_library

    include_module_interface_files = ctx.attr.include_module_interface_files
    include_resource_bundle = ctx.attr.include_resource_bundle

    target_os = _SDK_TO_OS[sdk]
    include_versioned_frameworks = ctx.attr.include_versioned_frameworks and target_os == "macos"

    if swift_library_files and len(architectures) > 1:
        fail("Internal error: Can only generate a Swift " +
             "framework with a single architecture at this time")

    headers = []
    module_interfaces = []

    if not swift_library_files:
        # Compile library
        binary = generation_support.compile_binary(
            actions = actions,
            apple_fragment = apple_fragment,
            archs = architectures,
            embed_bitcode = ctx.attr.embed_bitcode,
            embed_debug_info = ctx.attr.embed_debug_info,
            hdrs = hdrs,
            label = label,
            minimum_os_version = minimum_os_version,
            sdk = sdk,
            srcs = srcs,
            xcode_config = xcode_config,
        )

        # Create dynamic or static library
        if libtype == "dynamic":
            library = generation_support.create_dynamic_library(
                actions = actions,
                apple_fragment = apple_fragment,
                archs = architectures,
                binary = binary,
                label = label,
                minimum_os_version = minimum_os_version,
                sdk = sdk,
                xcode_config = xcode_config,
            )
        else:
            library = generation_support.create_static_library(
                actions = actions,
                apple_fragment = apple_fragment,
                binary = binary,
                label = label,
                xcode_config = xcode_config,
            )

        # Add headers to framework
        headers.extend(hdrs)

    else:
        # Get static library and generated header files
        library = generation_support.get_file_with_extension(
            files = swift_library_files,
            extension = "a",
        )
        headers.append(
            generation_support.get_file_with_extension(
                files = swift_library_files,
                extension = "h",
            ),
        )

        if include_module_interface_files:
            swiftinterface = generation_support.get_file_with_extension(
                files = swift_library_files,
                extension = "swiftinterface",
            )

            if swiftinterface:
                module_interfaces.extend([
                    generation_support.copy_file(
                        actions = actions,
                        file = swiftinterface,
                        label = label,
                        target_filename = "%s.swiftinterface" % architectures[0],
                    ),
                    generation_support.copy_file(
                        actions = actions,
                        file = swiftinterface,
                        label = label,
                        target_filename = "{arch}-apple-{os}{environment}.swiftinterface".format(
                            arch = architectures[0],
                            os = target_os,
                            environment = "-simulator" if sdk.endswith("simulator") else "",
                        ),
                    ),
                ])
            else:
                swiftmodule = generation_support.get_file_with_extension(
                    files = swift_library_files,
                    extension = "swiftmodule",
                )
                module_interfaces.append(
                    generation_support.copy_file(
                        actions = actions,
                        file = swiftmodule,
                        label = label,
                        target_filename = "%s.swiftmodule" % architectures[0],
                    ),
                )

    # Create framework bundle
    framework_files = generation_support.create_framework(
        actions = actions,
        apple_fragment = apple_fragment,
        bundle_name = label.name,
        headers = headers,
        include_resource_bundle = include_resource_bundle,
        include_versioned_frameworks = include_versioned_frameworks,
        is_dynamic = libtype == "dynamic",
        label = label,
        library = library,
        module_interfaces = module_interfaces,
        target_os = target_os,
        xcode_config = xcode_config,
    )

    return [
        DefaultInfo(files = depset(framework_files)),
    ]

generate_import_framework = rule(
    implementation = _generate_import_framework_impl,
    attrs = dicts.add(apple_support.action_required_attrs(), {
        "archs": attr.string_list(
            allow_empty = False,
            doc = "A list of architectures this framework will be generated for.",
        ),
        "sdk": attr.string(
            doc = """
Determines what SDK the framework will be built under.
""",
            values = [
                "appletvos",
                "appletvsimulator",
                "iphoneos",
                "iphonesimulator",
                "macosx",
                "watchos",
                "watchsimulator",
            ],
        ),
        "minimum_os_version": attr.string(
            doc = """
Minimum version of the OS corresponding to the SDK that this binary will support.
""",
        ),
        "src": attr.label(
            allow_single_file = True,
            default = Label(
                "//test/starlark_tests/resources/frameworks:SharedClass.m",
            ),
            doc = "Source file for the generated framework.",
        ),
        "hdrs": attr.label(
            allow_files = True,
            default = Label(
                "//test/starlark_tests/resources/frameworks:SharedClass.h",
            ),
            doc = "Header files for the generated framework.",
        ),
        "include_resource_bundle": attr.bool(
            mandatory = False,
            default = False,
            doc = """
Boolean to indicate if the generate framework should include a resource bundle containing an
Info.plist file to test resource propagation.
""",
        ),
        "swift_library": attr.label(
            allow_files = True,
            doc = "Label for a Swift library target to source archive and module files from.",
            providers = [SwiftInfo],
        ),
        "libtype": attr.string(
            values = ["dynamic", "static"],
            doc = """
Possible values are `dynamic` or `static`.
Determines if the framework will be built as a dynamic framework or a static framework.
""",
        ),
        "embed_bitcode": attr.bool(
            default = False,
            doc = """
Set to `True` to generate and embed bitcode in the final framework binary.
""",
        ),
        "embed_debug_info": attr.bool(
            default = False,
            doc = """
Set to `True` to generate and embed debug information in the framework
binary.
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
        "include_versioned_frameworks": attr.bool(
            default = True,
            doc = """
Flag to indicate if the framework should include additional versions of the framework under the
Versions directory. This is only supported for macOS platform.
                """,
        ),
    }),
    fragments = ["apple"],
    doc = """
Generates an imported dynamic framework for testing.

Provides:
  A dynamic framework target that can be referenced through an apple_*_framework_import rule.
""",
)
