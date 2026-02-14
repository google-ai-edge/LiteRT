# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Defines rules for building Apple DocC targets."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple:providers.bzl",
    "DocCBundleInfo",
    "DocCSymbolGraphsInfo",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebinaryinfo",
)
load(
    "//apple/internal/aspects:docc_archive_aspect.bzl",
    "docc_bundle_info_aspect",
    "docc_symbol_graphs_aspect",
)

def _docc_archive_impl(ctx):
    """Builds a .doccarchive for the given module.
    """

    apple_fragment = ctx.fragments.apple
    default_code_listing_language = ctx.attr.default_code_listing_language
    diagnostic_level = ctx.attr.diagnostic_level
    enable_inherited_docs = ctx.attr.enable_inherited_docs
    execution_requirements = {}
    fallback_bundle_identifier = ctx.attr.fallback_bundle_identifier
    fallback_bundle_version = ctx.attr.fallback_bundle_version
    fallback_display_name = ctx.attr.fallback_display_name
    hosting_base_path = ctx.attr.hosting_base_path
    kinds = ctx.attr.kinds
    platform = ctx.fragments.apple.single_arch_platform
    transform_for_static_hosting = ctx.attr.transform_for_static_hosting
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    dep = ctx.attr.dep
    symbol_graphs_info = None
    docc_bundle_info = None
    docc_build_inputs = []

    if DocCSymbolGraphsInfo in dep:
        symbol_graphs_info = dep[DocCSymbolGraphsInfo]
    if DocCBundleInfo in dep:
        docc_bundle_info = dep[DocCBundleInfo]

    if not symbol_graphs_info and not docc_bundle_info:
        fail("At least one of DocCSymbolGraphsInfo or DocCBundleInfo must be provided for target %s" % ctx.attr.name)

    symbol_graphs = symbol_graphs_info.symbol_graphs.to_list() if symbol_graphs_info else []

    if ctx.attr.name.endswith(".doccarchive"):
        doccarchive_dir = ctx.actions.declare_directory(ctx.attr.name)
    else:
        doccarchive_dir = ctx.actions.declare_directory("%s.doccarchive" % ctx.attr.name)

    # Command and required arguments
    arguments = ctx.actions.args()
    arguments.add("docc")
    arguments.add("convert")
    arguments.add("--fallback-display-name", fallback_display_name)
    arguments.add("--fallback-bundle-identifier", fallback_bundle_identifier)
    arguments.add("--fallback-bundle-version", fallback_bundle_version)
    arguments.add("--output-dir", doccarchive_dir.path)

    # Optional agruments
    if default_code_listing_language:
        arguments.add("--default-code-listing-language", default_code_listing_language)
    if diagnostic_level:
        arguments.add("--diagnostic-level", diagnostic_level)
    if enable_inherited_docs:
        arguments.add("--enable-inherited-docs")
    if kinds:
        arguments.add_all("--kind", kinds)
    if transform_for_static_hosting:
        arguments.add("--transform-for-static-hosting")
    if hosting_base_path:
        arguments.add("--hosting-base-path", hosting_base_path)

    # Add symbol graphs
    if symbol_graphs_info:
        arguments.add_all(
            symbol_graphs,
            before_each = "--additional-symbol-graph-dir",
            expand_directories = False,
        )
        docc_build_inputs.extend(symbol_graphs)

    # The .docc bundle (if provided, only one is allowed)
    if docc_bundle_info:
        arguments.add(docc_bundle_info.bundle)

        # TODO: no-sandbox seems to be required when running docc convert with a .docc bundle provided
        # in the sandbox the tool is unable to open the .docc bundle.
        execution_requirements["no-sandbox"] = "1"
        docc_build_inputs.extend(docc_bundle_info.bundle_files)

    apple_support.run(
        actions = ctx.actions,
        xcode_config = xcode_config,
        apple_fragment = apple_fragment,
        inputs = depset(docc_build_inputs),
        outputs = [doccarchive_dir],
        mnemonic = "DocCConvert",
        executable = "/usr/bin/xcrun",
        arguments = [arguments],
        progress_message = "Converting .doccarchive for %{label}",
        execution_requirements = execution_requirements,
    )

    # Create an executable shell script that runs `docc preview` on the .doccarchive.
    preview_script = ctx.actions.declare_file("%s_preview.sh" % ctx.attr.name)
    ctx.actions.expand_template(
        output = preview_script,
        template = ctx.file._preview_template,
        substitutions = {
            "{docc_bundle}": docc_bundle_info.bundle if docc_bundle_info else "",
            "{fallback_bundle_identifier}": fallback_bundle_identifier,
            "{fallback_bundle_version}": str(fallback_bundle_version),
            "{fallback_display_name}": fallback_display_name,
            "{platform}": platform.name_in_plist,
            "{sdk_version}": str(xcode_config.sdk_version_for_platform(platform)),
            "{symbol_graph_dirs}": " ".join([f.path for f in symbol_graphs]) if symbol_graphs else "",
            "{target_name}": ctx.attr.name,
            "{xcode_version}": str(xcode_config.xcode_version()),
        },
        is_executable = True,
    )

    # Limiting the contents of AppleBinaryInfo to what is necessary for testing and validation.
    doccarchive_binary_info = new_applebinaryinfo(
        binary = doccarchive_dir,
        infoplist = None,
        product_type = None,
    )

    return [
        DefaultInfo(
            files = depset([doccarchive_dir]),
            executable = preview_script,
            runfiles = ctx.runfiles(files = [
                preview_script,
            ] + docc_build_inputs),
        ),
        doccarchive_binary_info,
    ]

docc_archive = rule(
    implementation = _docc_archive_impl,
    fragments = ["apple"],
    doc = """
Builds a .doccarchive for the given dependency.
The target created by this rule can also be `run` to preview the generated documentation in Xcode.

NOTE: At this time Swift is the only supported language for this rule.

Example:

```starlark
load("@build_bazel_rules_apple//apple:docc.bzl", "docc_archive")

docc_archive(
    name = "Lib.doccarchive",
    dep = ":Lib",
    fallback_bundle_identifier = "com.example.lib",
    fallback_bundle_version = "1.0.0",
    fallback_display_name = "Lib",
)
```""",
    attrs = dicts.add(
        apple_support.action_required_attrs(),
        {
            "dep": attr.label(
                aspects = [
                    docc_bundle_info_aspect,
                    docc_symbol_graphs_aspect,
                ],
                providers = [[DocCBundleInfo], [DocCSymbolGraphsInfo]],
            ),
            "default_code_listing_language": attr.string(
                doc = "A fallback default language for code listings if no value is provided in the documentation bundle's Info.plist file.",
            ),
            "diagnostic_level": attr.string(
                doc = """
Filters diagnostics above this level from output
This filter level is inclusive. If a level of `information` is specified, diagnostics with a severity up to and including `information` will be printed.
Must be one of "error", "warning", "information", or "hint"
                """,
                values = ["error", "warning", "information", "hint"],
            ),
            # TODO: use `attr.bool` once https://github.com/bazelbuild/bazel/issues/22809 is resolved.
            "emit_extension_block_symbols": attr.string(
                default = "0",
                doc = """
Defines if the symbol graph information for `extension` blocks should be
emitted in addition to the default symbol graph information.

This value must be either `"0"` or `"1"`.When the value is `"1"`, the symbol
graph information for `extension` blocks will be emitted in addition to
the default symbol graph information. The default value is `"0"`.
                """,
                values = ["0", "1"],
            ),
            "enable_inherited_docs": attr.bool(
                default = False,
                doc = "Inherit documentation for inherited symbols.",
            ),
            "fallback_bundle_identifier": attr.string(
                doc = "A fallback bundle identifier if no value is provided in the documentation bundle's Info.plist file.",
                mandatory = True,
            ),
            "fallback_bundle_version": attr.string(
                doc = "A fallback bundle version if no value is provided in the documentation bundle's Info.plist file.",
                mandatory = True,
            ),
            "fallback_display_name": attr.string(
                doc = "A fallback display name if no value is provided in the documentation bundle's Info.plist file.",
                mandatory = True,
            ),
            "hosting_base_path": attr.string(
                doc = "The base path your documentation website will be hosted at. For example, to deploy your site to 'example.com/my_name/my_project/documentation' instead of 'example.com/documentation', pass '/my_name/my_project' as the base path.",
                mandatory = False,
            ),
            "kinds": attr.string_list(
                doc = "The kinds of entities to filter generated documentation for.",
            ),
            "minimum_access_level": attr.string(
                default = "public",
                doc = """"
The minimum access level of the declarations that should be emitted in the symbol graphs.
This value must be either `fileprivate`, `internal`, `private`, or `public`. The default value is `public`.
                """,
                values = [
                    "fileprivate",
                    "internal",
                    "private",
                    "public",
                ],
            ),
            "transform_for_static_hosting": attr.bool(
                default = True,
            ),
            "_preview_template": attr.label(
                allow_single_file = True,
                default = "//apple/internal/templates:docc_preview_template",
            ),
        },
    ),
    executable = True,
)
