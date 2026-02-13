# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""Shared toolchain required for processing Apple bundling rules."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

AppleMacToolsToolchainInfo = provider(
    doc = """
Propagates information about an Apple toolchain to internal bundling rules that use the toolchain.

This provider exists as an internal detail for the rules to reference common, executable tools and
files used as script templates for the purposes of executing Apple actions. Defined by the
`apple_mac_tools_toolchain` rule.

This toolchain is for the tools (and support files) for actions that *must* run on a Mac.
""",
    fields = {
        "dsym_info_plist_template": """\
A `File` referencing a plist template for dSYM bundles.
""",
        "process_and_sign_template": """\
A `File` referencing a template for a shell script to process and sign.
""",
        "alticonstool": """\
The files_to_run for a tool to insert alternate icons entries in the app
bundle's `Info.plist`.
""",
        "bundletool_experimental": """\
The files_to_run for an experimental tool to create an Apple bundle by
combining the bundling, post-processing, and signing steps into a single action that eliminates the
archiving step.
""",
        "clangrttool": """\
The files_to_run for a tool to find all Clang runtime libs linked to a
binary.
""",
        "codesigningtool": """\
The files_to_run for a tool to select the appropriate signing identity
for Apple apps and Apple executable bundles.
""",
        "dossier_codesigningtool": """\
The files_to_run for a tool to generate codesigning dossiers.
""",
        "environment_plist_tool": """\
The files_to_run for a tool for collecting dev environment values.
""",
        "imported_dynamic_framework_processor": """\
The files_to_run for a tool to process an imported dynamic framework
such that the given framework only contains the same slices as the app binary, every file belonging
to the dynamic framework is copied to a temporary location, and the dynamic framework is codesigned
and zipped as a cacheable artifact.
""",
        "main_thread_checker_tool": """\
The files_to_run for a tool to find libMainThreadChecker.dylib linked to a
binary.
""",
        "plisttool": """\
The files_to_run for a tool to perform plist operations such as variable
substitution, merging, and conversion of plist files to binary format.
""",
        "provisioning_profile_tool": """\
The files_to_run for a tool that extracts entitlements from a
provisioning profile.
""",
        "swift_stdlib_tool": """\
The files_to_run for a tool that copies and lipos Swift stdlibs required
for the target to run.
""",
        "xcframework_processor_tool": """\
The files_to_run for a tool that extracts and copies an XCFramework
library for a target triplet.
""",
        "xctoolrunner": """\
The files_to_run for a tool that acts as a wrapper for xcrun actions.
""",
    },
)

AppleXPlatToolsToolchainInfo = provider(
    doc = """
Propagates information about an Apple toolchain to internal bundling rules that use the toolchain.

This provider exists as an internal detail for the rules to reference common, executable tools and
files used as script templates for the purposes of executing Apple actions. Defined by the
`apple_xplat_tools_toolchain` rule.

This toolchain is for the tools (and support files) for actions that can run on any platform,
i.e. - they do *not* have to run on a Mac.
""",
    fields = {
        "build_settings": """\
A `struct` containing custom build settings values, where fields are the name of the build setting
target name and values are retrieved from the BuildSettingInfo provider for each label provided.

e.g. apple_xplat_tools_toolchaininfo.build_settings.signing_certificate_name
""",
        "bundletool": """\
A files_to_run for a tool to create an Apple bundle by taking a list of
files/ZIPs and destinations paths to build the directory structure for those files.
""",
        "versiontool": """\
A files_to_run for a tool that acts as a wrapper for xcrun actions.
""",
    },
)

def _shared_attrs():
    """Private attributes on every rule to provide access to bundling tools and other file deps."""
    return {
        "_mac_toolchain": attr.label(
            default = Label("//apple/internal:mac_tools_toolchain"),
            providers = [[AppleMacToolsToolchainInfo]],
            cfg = "exec",
        ),
        "_xplat_toolchain": attr.label(
            default = Label("//apple/internal:xplat_tools_toolchain"),
            providers = [[AppleXPlatToolsToolchainInfo]],
            cfg = "exec",
        ),
    }

def _apple_mac_tools_toolchain_impl(ctx):
    return [
        AppleMacToolsToolchainInfo(
            dsym_info_plist_template = ctx.file.dsym_info_plist_template,
            process_and_sign_template = ctx.file.process_and_sign_template,
            alticonstool = ctx.attr.alticonstool.files_to_run,
            bundletool_experimental = ctx.attr.bundletool_experimental.files_to_run,
            codesigningtool = ctx.attr.codesigningtool.files_to_run,
            dossier_codesigningtool = ctx.attr.dossier_codesigningtool.files_to_run,
            clangrttool = ctx.attr.clangrttool.files_to_run,
            main_thread_checker_tool = ctx.attr.main_thread_checker_tool.files_to_run,
            environment_plist_tool = ctx.attr.environment_plist_tool.files_to_run,
            imported_dynamic_framework_processor = ctx.attr.imported_dynamic_framework_processor.files_to_run,
            plisttool = ctx.attr.plisttool.files_to_run,
            provisioning_profile_tool = ctx.attr.provisioning_profile_tool.files_to_run,
            swift_stdlib_tool = ctx.attr.swift_stdlib_tool.files_to_run,
            xcframework_processor_tool = ctx.attr.xcframework_processor_tool.files_to_run,
            xctoolrunner = ctx.attr.xctoolrunner.files_to_run,
        ),
        DefaultInfo(),
    ]

apple_mac_tools_toolchain = rule(
    attrs = {
        "alticonstool": attr.label(
            cfg = "exec",
            executable = True,
            doc = """
A `File` referencing a tool to insert alternate icons entries in the app bundle's `Info.plist`.
""",
        ),
        "bundletool_experimental": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing an experimental tool to create an Apple bundle by combining the bundling,
post-processing, and signing steps into a single action that eliminates the archiving step.
""",
        ),
        "clangrttool": attr.label(
            cfg = "target",
            executable = True,
            doc = "A `File` referencing a tool to find all Clang runtime libs linked to a binary.",
        ),
        "codesigningtool": attr.label(
            cfg = "target",
            executable = True,
            doc = "A `File` referencing a tool to assist in signing bundles.",
        ),
        "dossier_codesigningtool": attr.label(
            cfg = "target",
            executable = True,
            doc = "A `File` referencing a tool to assist in generating signing dossiers.",
        ),
        "dsym_info_plist_template": attr.label(
            cfg = "target",
            allow_single_file = True,
            doc = "A `File` referencing a plist template for dSYM bundles.",
        ),
        "environment_plist_tool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool to collect data from the development environment to be record into
final bundles.
""",
        ),
        "imported_dynamic_framework_processor": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool to process an imported dynamic framework such that the given framework
only contains the same slices as the app binary, every file belonging to the dynamic framework is
copied to a temporary location, and the dynamic framework is codesigned and zipped as a cacheable
artifact.
""",
        ),
        "main_thread_checker_tool": attr.label(
            cfg = "target",
            executable = True,
            doc = "A `File` referencing a tool to find libMainThreadChecker.dylib linked to a binary.",
        ),
        "plisttool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool to perform plist operations such as variable substitution, merging, and
conversion of plist files to binary format.
""",
        ),
        "process_and_sign_template": attr.label(
            allow_single_file = True,
            doc = "A `File` referencing a template for a shell script to process and sign.",
        ),
        "provisioning_profile_tool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool that extracts entitlements from a provisioning profile.
""",
        ),
        "swift_stdlib_tool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool that copies and lipos Swift stdlibs required for the target to run.
""",
        ),
        "xcframework_processor_tool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool that extracts and copies an XCFramework library for a given target
triplet.
""",
        ),
        "xctoolrunner": attr.label(
            cfg = "target",
            executable = True,
            doc = "A `File` referencing a tool that acts as a wrapper for xcrun actions.",
        ),
    },
    doc = """Represents an Apple support toolchain for tools that must run on a Mac""",
    implementation = _apple_mac_tools_toolchain_impl,
)

def _apple_xplat_tools_toolchain_impl(ctx):
    return [
        AppleXPlatToolsToolchainInfo(
            build_settings = struct(
                **{
                    build_setting.label.name: build_setting[BuildSettingInfo].value
                    for build_setting in ctx.attr.build_settings
                }
            ),
            bundletool = ctx.attr.bundletool.files_to_run,
            versiontool = ctx.attr.versiontool.files_to_run,
        ),
        DefaultInfo(),
    ]

apple_xplat_tools_toolchain = rule(
    attrs = {
        "build_settings": attr.label_list(
            providers = [BuildSettingInfo],
            mandatory = True,
            doc = """
List of `Label`s referencing custom build settings for all Apple rules.
""",
        ),
        "bundletool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool to create an Apple bundle by taking a list of files/ZIPs and destination
paths to build the directory structure for those files.
""",
        ),
        "versiontool": attr.label(
            cfg = "target",
            executable = True,
            doc = """
A `File` referencing a tool for extracting version info from builds.
""",
        ),
    },
    doc = """Represents an Apple support toolchain for tools that can run on any platform""",
    implementation = _apple_xplat_tools_toolchain_impl,
)

# Define the loadable module that lists the exported symbols in this file.
apple_toolchain_utils = struct(
    shared_attrs = _shared_attrs,
)
