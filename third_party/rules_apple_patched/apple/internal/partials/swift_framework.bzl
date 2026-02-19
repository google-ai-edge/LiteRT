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

"""Partial implementation for Swift frameworks with third party interfaces."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)
load(
    "//apple/internal:swift_info_support.bzl",
    "swift_info_support",
)

def _swift_framework_partial_impl(
        *,
        actions,
        avoid_deps,
        bundle_name,
        framework_modulemap,
        label_name,
        output_discriminator,
        swift_infos):
    """Implementation for the Swift framework processing partial."""

    if len(swift_infos) == 0:
        fail("""
Internal error: Expected to find a SwiftInfo before entering this partial. Please file an \
issue with a reproducible error case.
""")

    avoid_modules = swift_info_support.modules_from_avoid_deps(avoid_deps = avoid_deps)
    bundle_files = []
    expected_module_name = bundle_name
    found_module_name = None
    found_generated_header = None
    modules_parent = paths.join("Modules", "{}.swiftmodule".format(expected_module_name))

    for arch, swiftinfo in swift_infos.items():
        swift_module = swift_info_support.swift_include_info(
            avoid_modules = avoid_modules,
            found_module_name = found_module_name,
            transitive_modules = swiftinfo.transitive_modules,
        )

        # If headers are generated, they should be generated equally for all archs, so just take the
        # first one found.
        if (not found_generated_header) and swift_module.clang:
            if swift_module.clang.compilation_context.direct_headers:
                found_generated_header = swift_module.clang.compilation_context.direct_headers[0]

        found_module_name = swift_module.name

        # A swiftinterface will not be present when library evolution is disabled, if so, fallback to swiftmodule.
        if swift_module.swift.swiftinterface:
            bundle_interface = swift_info_support.declare_swiftinterface(
                actions = actions,
                arch = arch,
                label_name = label_name,
                output_discriminator = output_discriminator,
                swiftinterface = swift_module.swift.swiftinterface,
            )
            bundle_files.append((processor.location.bundle, modules_parent, depset([bundle_interface])))
        else:
            bundle_swiftmodule = swift_info_support.declare_swiftmodule(
                actions = actions,
                arch = arch,
                label_name = label_name,
                output_discriminator = output_discriminator,
                swiftmodule = swift_module.swift.swiftmodule,
            )
            bundle_files.append((processor.location.bundle, modules_parent, depset([bundle_swiftmodule])))

        bundle_doc = swift_info_support.declare_swiftdoc(
            actions = actions,
            arch = arch,
            label_name = label_name,
            output_discriminator = output_discriminator,
            swiftdoc = swift_module.swift.swiftdoc,
        )
        bundle_files.append((processor.location.bundle, modules_parent, depset([bundle_doc])))

    swift_info_support.verify_found_module_name(
        bundle_name = expected_module_name,
        found_module_name = found_module_name,
    )

    if found_generated_header:
        bundle_header = swift_info_support.declare_generated_header(
            actions = actions,
            generated_header = found_generated_header,
            label_name = label_name,
            output_discriminator = output_discriminator,
            module_name = expected_module_name,
        )
        bundle_files.append((processor.location.bundle, "Headers", depset([bundle_header])))

        modulemap = swift_info_support.declare_modulemap(
            actions = actions,
            framework_modulemap = framework_modulemap,
            label_name = label_name,
            output_discriminator = output_discriminator,
            module_name = expected_module_name,
        )
        bundle_files.append((processor.location.bundle, "Modules", depset([modulemap])))

    return struct(bundle_files = bundle_files)

def swift_framework_partial(
        *,
        actions,
        avoid_deps = [],
        bundle_name,
        framework_modulemap = True,
        label_name,
        output_discriminator = None,
        swift_infos):
    """Constructor for the Swift framework processing partial.

    This partial collects and bundles the necessary files to construct a Swift based static
    framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        avoid_deps: A list of library targets with modules to avoid, if specified.
        bundle_name: The name of the output bundle.
        framework_modulemap: Boolean to indicate if the generated modulemap should be for a
            framework instead of a library or a generic module. Defaults to `True`.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        swift_infos: A dictionary with architectures as keys and the SwiftInfo provider containing
            the required artifacts for that architecture as values.

    Returns:
        A partial that returns the bundle location of the supporting Swift artifacts needed in a
        Swift based sdk framework.
    """
    return partial.make(
        _swift_framework_partial_impl,
        actions = actions,
        avoid_deps = avoid_deps,
        bundle_name = bundle_name,
        framework_modulemap = framework_modulemap,
        label_name = label_name,
        output_discriminator = output_discriminator,
        swift_infos = swift_infos,
    )
