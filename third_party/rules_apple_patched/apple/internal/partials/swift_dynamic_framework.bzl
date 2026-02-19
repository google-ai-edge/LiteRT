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

"""Partial implementation for Swift dynamic frameworks."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)

# TODO(b/161370390): Remove ctx from the args when ctx is removed from all partials.
def _swift_dynamic_framework_partial_impl(
        *,
        actions,
        bundle_name,
        label_name,
        swift_dynamic_framework_info):
    """Implementation for the Swift dynamic framework processing partial."""

    if bundle_name != swift_dynamic_framework_info.module_name:
        fail("""
error: Found swift_library with module name {} but expected {}. Swift dynamic \
frameworks expect a single swift_library dependency with `module_name` set to the same \
`bundle_name` as the dynamic framework target.\
""".format(swift_dynamic_framework_info.module_name, bundle_name))

    generated_header = swift_dynamic_framework_info.generated_header
    swiftdocs = swift_dynamic_framework_info.swiftdocs
    swiftmodules = swift_dynamic_framework_info.swiftmodules
    modulemap_file = swift_dynamic_framework_info.modulemap

    bundle_files = []
    modules_parent = paths.join("Modules", "{}.swiftmodule".format(bundle_name))

    for arch, swiftdoc in swiftdocs.items():
        bundle_doc = intermediates.file(
            actions = actions,
            file_name = "{}.swiftdoc".format(arch),
            output_discriminator = None,
            target_name = label_name,
        )
        actions.symlink(target_file = swiftdoc, output = bundle_doc)
        bundle_files.append((processor.location.bundle, modules_parent, depset([bundle_doc])))

    for arch, swiftmodule in swiftmodules.items():
        bundle_doc = intermediates.file(
            actions = actions,
            file_name = "{}.swiftmodule".format(arch),
            output_discriminator = None,
            target_name = label_name,
        )
        actions.symlink(target_file = swiftmodule, output = bundle_doc)
        bundle_files.append((processor.location.bundle, modules_parent, depset([bundle_doc])))

    if generated_header:
        bundle_header = intermediates.file(
            actions = actions,
            file_name = "{}.h".format(bundle_name),
            output_discriminator = None,
            target_name = label_name,
        )
        actions.symlink(target_file = generated_header, output = bundle_header)
        bundle_files.append((processor.location.bundle, "Headers", depset([bundle_header])))

    if modulemap_file:
        modulemap = intermediates.file(
            actions = actions,
            file_name = "module.modulemap",
            output_discriminator = None,
            target_name = label_name,
        )
        actions.symlink(target_file = modulemap_file, output = modulemap)
        bundle_files.append((processor.location.bundle, "Modules", depset([modulemap])))

    return struct(
        bundle_files = bundle_files,
    )

def swift_dynamic_framework_partial(
        *,
        actions,
        bundle_name,
        label_name,
        swift_dynamic_framework_info):
    """Constructor for the Swift dynamic framework processing partial.

    This partial collects and bundles the necessary files to construct a Swift based dynamic
    framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        bundle_name: The name of the output bundle.
        label_name: Name of the target being built.
        swift_dynamic_framework_info: The SwiftDynamicFrameworkInfo provider containing the required
            artifacts.

    Returns:
        A partial that returns the bundle location of the supporting Swift artifacts needed in a
        Swift based dynamic framework.
    """
    return partial.make(
        _swift_dynamic_framework_partial_impl,
        actions = actions,
        bundle_name = bundle_name,
        label_name = label_name,
        swift_dynamic_framework_info = swift_dynamic_framework_info,
    )
