# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""Implementation of Apple CoreML library rule."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
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
load(
    "//apple/internal:rule_attrs.bzl",
    "rule_attrs",
)
load(
    "//apple/internal:swift_support.bzl",
    "swift_support",
)

def _apple_core_ml_library_impl(ctx):
    """Implementation of the apple_core_ml_library."""
    actions = ctx.actions
    apple_xplat_toolchain_info = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo]
    basename = paths.replace_extension(ctx.file.mlmodel.basename, "")

    is_swift = ctx.attr.language == "Swift"

    deps = getattr(ctx.attr, "deps", None)
    uses_swift = is_swift or (swift_support.uses_swift(deps) if deps else False)

    if is_swift and not ctx.attr.swift_source:
        fail("Attribute `swift_source` is mandatory when generating Swift.")
    if not is_swift and (not ctx.attr.objc_header or not ctx.attr.objc_source):
        fail("Attributes `objc_header` and `objc_source` are mandatory when generating Objective-C.")

    swift_output_src = None
    objc_output_src = None
    objc_output_hdr = None

    if is_swift:
        swift_output_src = actions.declare_file("{}.swift".format(basename))
    else:
        objc_output_src = actions.declare_file("{}.m".format(basename))
        objc_output_hdr = actions.declare_file("{}.h".format(basename))

    features = features_support.compute_enabled_features(
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    # TODO(b/168721966): Consider if an aspect could be used to generate mlmodel sources. This
    # would be similar to how we are planning to use the resource aspect with the
    # apple_resource_bundle and apple_resource_group resource rules. That might allow for more
    # portable platform information.
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
        uses_swift = uses_swift,
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )

    apple_mac_toolchain_info = ctx.attr._mac_toolchain[AppleMacToolsToolchainInfo]

    # coremlc doesn't have any configuration on the name of the generated source files, it uses the
    # basename of the mlmodel file instead, so we need to expect those files as outputs.
    resource_actions.generate_mlmodel_sources(
        actions = actions,
        language = ctx.attr.language,
        input_file = ctx.file.mlmodel,
        swift_output_src = swift_output_src,
        objc_output_src = objc_output_src,
        objc_output_hdr = objc_output_hdr,
        platform_prerequisites = platform_prerequisites,
        xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
    )

    if is_swift:
        actions.symlink(target_file = swift_output_src, output = ctx.outputs.swift_source)
        return [
            DefaultInfo(files = depset([swift_output_src])),
        ]

    actions.write(
        ctx.outputs.objc_public_header,
        "#import \"{}\"".format(objc_output_hdr.path),
    )

    actions.symlink(target_file = objc_output_hdr, output = ctx.outputs.objc_header)
    actions.symlink(target_file = objc_output_src, output = ctx.outputs.objc_source)

    # This rule returns the headers as its outputs so that they can be referenced in the hdrs of the
    # underlying objc_library.
    return [
        DefaultInfo(
            files = depset([ctx.outputs.objc_public_header, objc_output_hdr, objc_output_src, ctx.outputs.objc_header, ctx.outputs.objc_source]),
        ),
    ]

apple_core_ml_library = rule(
    implementation = _apple_core_ml_library_impl,
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        apple_toolchain_utils.shared_attrs(),
        rule_attrs.common_tool_attrs(),
        {
            "mlmodel": attr.label(
                allow_single_file = ["mlmodel", "mlpackage"],
                mandatory = True,
                doc = """
Label to a single `.mlmodel` file or `.mlpackage` bundle from which to generate sources and compile
into mlmodelc files.
""",
            ),
            "language": attr.string(
                mandatory = True,
                values = ["Objective-C", "Swift"],
                doc = "Language of generated classes (\"Objective-C\", \"Swift\")",
            ),
            "swift_source": attr.output(
                doc = "Name of the output file (only when using Swift).",
            ),
            "objc_source": attr.output(
                doc = "Name of the implementation file (only when using Objective-C).",
            ),
            "objc_header": attr.output(
                doc = "Name of the header file (only when using Objective-C).",
            ),
            "objc_public_header": attr.output(
                doc = "Name of the public header file (only when using Objective-C).",
            ),
        },
    ),
    fragments = ["apple"],
    doc = """
This rule takes a single mlmodel file or mlpackage bundle and creates a target that can be added
as a dependency from objc_library or swift_library targets. For Swift, just import like any other
objc_library target. For objc_library, this target generates a header named `<target_name>.h` that
can be imported from within the package where this target resides. For example, if this target's
label is `//my/package:coreml`, you can import the header as `#import "my/package/coreml.h"`.

This rule supports the integration of CoreML `mlmodel` files into Apple rules.
`apple_core_ml_library` targets are added directly into `deps` for both
`objc_library` and `swift_library` targets.

For Swift, import the `apple_core_ml_library` the same way you'd import an
`objc_library` or `swift_library` target. For `objc_library` targets,
`apple_core_ml_library` creates a header file named after the target.

For example, if the `apple_core_ml_library` target's label is
`//my/package:MyModel`, then to import this module in Swift you need to use
`import my_package_MyModel`. From Objective-C sources, you'd import the header
as `#import my/package/MyModel.h`.

This rule will also compile the `mlmodel` into an `mlmodelc` and propagate it
upstream so that it is packaged as a resource inside the top level bundle.
""",
)
