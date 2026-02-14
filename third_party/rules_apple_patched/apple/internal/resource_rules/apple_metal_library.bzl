# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Implementation of Metal shader library rule."""

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
    "//apple/internal:resources.bzl",
    "resources",
)

def _metal_apple_target_triple(platform_prerequisites):
    """Returns a `metal` compiler-compatible target triple string.

    Args:
      platform_prerequisites: Struct containing information on the platform being targeted.

    Returns:
      A Metal target triple string for use with the `-target` copt.
    """
    target_os_version = platform_prerequisites.minimum_os

    platform = platform_prerequisites.apple_fragment.single_arch_platform
    platform_string = str(platform.platform_type)
    if platform_string == "macos":
        platform_string = "macosx"

    environment = "" if platform.is_device else "-simulator"

    return "air64-apple-{platform}{version}{environment}".format(
        environment = environment,
        platform = platform_string,
        version = target_os_version,
    )

def _apple_metal_library_impl(ctx):
    """Implementation of the apple_metal_library rule."""

    platform_prerequisites = platform_support.platform_prerequisites(
        apple_fragment = ctx.fragments.apple,
        build_settings = ctx.attr._xplat_toolchain[AppleXPlatToolsToolchainInfo].build_settings,
        config_vars = ctx.var,
        device_families = None,
        explicit_minimum_deployment_os = None,
        explicit_minimum_os = None,
        features = features_support.compute_enabled_features(
            requested_features = ctx.features,
            unsupported_features = ctx.disabled_features,
        ),
        objc_fragment = None,
        platform_type_string = str(ctx.fragments.apple.single_arch_platform.platform_type),
        uses_swift = False,
        xcode_version_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
    )

    target = _metal_apple_target_triple(platform_prerequisites)
    out = ctx.actions.declare_file(ctx.attr.out)

    args = ctx.actions.args()
    args.add("metal")
    args.add("-target", target)
    args.add("-o", out.path)
    args.add_all(ctx.attr.copts)
    args.add_all(ctx.files.srcs)

    apple_support.run(
        actions = ctx.actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        arguments = [args],
        executable = "/usr/bin/xcrun",
        inputs = ctx.files.srcs + ctx.files.hdrs,
        mnemonic = "MetallibCompile",
        outputs = [out],
        xcode_config = platform_prerequisites.xcode_version_config,
    )

    return [
        DefaultInfo(files = depset([out])),
        resources.bucketize_typed([out], "unprocessed"),
    ]

apple_metal_library = rule(
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        apple_toolchain_utils.shared_attrs(),
        {
            "copts": attr.string_list(
                doc = """\
A list of compiler options passed to the `metal` compiler for each source.
""",
            ),
            "hdrs": attr.label_list(
                allow_files = [".h"],
                doc = """\
A list of headers to make importable when compiling the metal library.
""",
            ),
            "includes": attr.string_list(
                doc = """\
A list of header search paths.
""",
            ),
            "out": attr.string(
                default = "default.metallib",
                doc = """\
An output `.metallib` filename. Defaults to `default.metallib` if unspecified.
""",
            ),
            "srcs": attr.label_list(
                allow_files = [".metal"],
                doc = """\
A list of `.metal` source files that will be compiled into the library.
""",
                mandatory = True,
            ),
        },
    ),
    doc = """
Compiles Metal shader language sources into a Metal library.
""",
    fragments = ["apple"],
    implementation = _apple_metal_library_impl,
)
