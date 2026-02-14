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

"""Rules to generate dSYM bundle from given framework for testing."""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple:utils.bzl",
    "group_files_by_directory",
)
load(
    ":generation_support.bzl",
    "generation_support",
)

def _get_framework_binary_file(framework_imports, framework_binary_path):
    for framework_import in framework_imports:
        if framework_import.path == framework_binary_path:
            return framework_import

    fail("Framework must contain a binary named after the framework")

def _generate_import_framework_dsym_impl(ctx):
    apple_fragment = ctx.fragments.apple
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    framework_imports = ctx.files.framework_imports
    framework_groups = group_files_by_directory(
        framework_imports,
        ["framework"],
        attr = "framework_imports",
    )

    framework_dir = framework_groups.keys()[0]
    framework_dir_name = paths.basename(framework_dir)
    framework_name = paths.split_extension(framework_dir_name)[0]
    framework_binary_path = paths.join(framework_dir, framework_name)

    framework_binary = _get_framework_binary_file(
        framework_imports,
        framework_binary_path,
    )

    dsym_files = generation_support.create_dsym(
        actions = ctx.actions,
        apple_fragment = apple_fragment,
        framework_binary = framework_binary,
        label = ctx.label,
        xcode_config = xcode_config,
    )

    return [
        DefaultInfo(
            files = depset(dsym_files),
        ),
    ]

generate_import_framework_dsym = rule(
    implementation = _generate_import_framework_dsym_impl,
    attrs = dicts.add(apple_support.action_required_attrs(), {
        "framework_imports": attr.label_list(
            allow_files = True,
            doc = "The list of files under a `.framework` directory.",
        ),
    }),
    fragments = ["apple"],
    doc = """
Generates a dSYM bundle from given framework for testing. NOTE: The generated
dSYM's DWARF binary doesn't actually contain any debug symbol.
""",
)
