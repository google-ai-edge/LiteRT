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

"""Implementation of ObjC/Swift Intent library rule."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple/internal:apple_toolchains.bzl",
    "AppleMacToolsToolchainInfo",
    "AppleXPlatToolsToolchainInfo",
    "apple_toolchain_utils",
)
load(
    "//apple/internal:features_support.bzl",
    "features_support",
)
load(
    "//apple/internal:platform_support.bzl",
    "platform_support",
)
load(
    "//apple/internal:resource_actions.bzl",
    "resource_actions",
)

def _apple_intent_library_impl(ctx):
    """Implementation of the apple_intent_library."""

    is_swift = ctx.attr.language == "Swift"

    if not is_swift and not ctx.attr.header_name:
        fail("A public header name is mandatory when generating Objective-C.")

    swift_output_src = None
    objc_output_srcs = None
    objc_output_hdrs = None
    objc_public_header = None

    if is_swift:
        swift_output_src = ctx.actions.declare_file("{}.swift".format(ctx.attr.name))
    else:
        objc_output_srcs = ctx.actions.declare_directory("{}.srcs.m".format(ctx.attr.name))
        objc_output_hdrs = ctx.actions.declare_directory("{}.hdrs.h".format(ctx.attr.name))
        objc_public_header = ctx.actions.declare_file("{}.h".format(ctx.attr.header_name))

    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    platform_prerequisites = platform_support.platform_prerequisites(
        apple_fragment = ctx.fragments.apple,
        build_settings = apple_xplat_toolchain_info.build_settings,
        config_vars = ctx.var,
        device_families = None,
        explicit_minimum_deployment_os = None,
        explicit_minimum_os = None,
        features = features,
        objc_fragment = None,
        platform_type_string = str(ctx.fragments.apple.single_arch_platform.platform_type),
        uses_swift = False,
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )

    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]
    resource_actions.generate_intent_classes_sources(
        actions = ctx.actions,
        input_file = ctx.file.src,
        swift_output_src = swift_output_src,
        objc_output_srcs = objc_output_srcs,
        objc_output_hdrs = objc_output_hdrs,
        objc_public_header = objc_public_header,
        language = ctx.attr.language,
        class_prefix = ctx.attr.class_prefix,
        swift_version = ctx.attr.swift_version,
        class_visibility = ctx.attr.class_visibility,
        platform_prerequisites = platform_prerequisites,
        xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
    )

    if is_swift:
        return [
            DefaultInfo(files = depset([swift_output_src])),
        ]

    return [
        DefaultInfo(
            files = depset([objc_output_srcs, objc_output_hdrs, objc_public_header]),
        ),
        OutputGroupInfo(
            srcs = depset([objc_output_srcs]),
            hdrs = depset([objc_output_hdrs, objc_public_header]),
        ),
    ]

apple_intent_library = rule(
    implementation = _apple_intent_library_impl,
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        apple_toolchain_utils.shared_attrs(),
        {
            "src": attr.label(
                allow_single_file = [".intentdefinition"],
                mandatory = True,
                doc = """
Label to a single `.intentdefinition` files from which to generate sources files.
""",
            ),
            "language": attr.string(
                mandatory = True,
                values = ["Objective-C", "Swift"],
                doc = "Language of generated classes (\"Objective-C\", \"Swift\")",
            ),
            "class_prefix": attr.string(
                doc = "Class prefix to use for the generated classes.",
            ),
            "swift_version": attr.string(
                doc = "Version of Swift to use for the generated classes.",
            ),
            "class_visibility": attr.string(
                values = ["public", "private", "project"],
                default = "",
                doc = "Visibility attribute for the generated classes (\"public\", \"private\", \"project\").",
            ),
            "header_name": attr.string(
                doc = "Name of the public header file (only when using Objective-C).",
            ),
        },
    ),
    fragments = ["apple"],
    doc = """
This rule supports the integration of Intents `.intentdefinition` files into Apple rules.
It takes a single `.intentdefinition` file and creates a target that can be added as a dependency from `objc_library` or
`swift_library` targets. It accepts the regular `objc_library` attributes too.
This target generates a header named `<target_name>.h` that can be imported from within the package where this target
resides. For example, if this target's label is `//my/package:intent`, you can import the header as
`#import "my/package/intent.h"`.
""",
)
