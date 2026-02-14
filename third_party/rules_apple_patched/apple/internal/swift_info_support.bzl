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

"""Support methods for handling artifacts from SwiftInfo providers."""

load("@bazel_skylib//lib:sets.bzl", "sets")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)

def _verify_found_module_name(*, bundle_name, found_module_name):
    """Validate that the module name fits the requirements for Swift frameworks.

    Args:
        bundle_name: The bundle name for this Swift framework.
        found_module_name: The module name that was found from a SwiftInfo provider.
    """
    if bundle_name != found_module_name:
        fail("""
error: Found swift_library with module name {actual} but expected {expected}. Swift static \
frameworks expect a single swift_library dependency with `module_name` set to the same \
`bundle_name` as the static framework target.\
""".format(
            actual = found_module_name,
            expected = bundle_name,
        ))

def _modules_from_avoid_deps(*, avoid_deps):
    """Returns a set of module names found from the SwiftInfo providers of avoid_deps"""
    avoid_swiftinfos = [t[SwiftInfo] for t in avoid_deps if SwiftInfo in t]
    avoid_modules = sets.make()
    for swiftinfo in avoid_swiftinfos:
        for module in swiftinfo.transitive_modules.to_list():
            if not module.swift:
                continue
            sets.insert(avoid_modules, module.name)
    return avoid_modules

def _swift_include_info(
        *,
        avoid_modules = sets.make(),
        found_module_name,
        transitive_modules):
    """Returns the module containing the Swift interface information from a SwiftInfo provider.

    Args:
        avoid_modules: A set of modules to avoid, if specified.
        found_module_name: The module name that was previously found from transitive deps.
        transitive_modules: The transitive_modules field of a SwiftInfo provider.

    Returns:
        The module found from `transitive_modules` that has the necessary swift interfaces.
    """
    swift_module = None
    transitive_modules_list = transitive_modules.to_list()

    for module in transitive_modules_list:
        if not module.swift or sets.contains(avoid_modules, module.name):
            continue

        if swift_module or (found_module_name and module.name != found_module_name):
            fail(
                """\
Error: Swift third party frameworks expect a single swift_library dependency with no transitive \
swift_library dependencies.\
""",
            )

        if not all([module.name, module.swift.swiftdoc]) or not (module.swift.swiftmodule or module.swift.swiftinterface):
            fail(
                """\
Error: Could not find all required artifacts and information to build a Swift framework. \
Please file an issue with a reproducible error case.\
""",
            )

        swift_module = module

    if not swift_module:
        if not transitive_modules_list:
            fail("""\
Internal Error: Swift third party frameworks require a Swift module to be defined from a \
"swift_library", but could not find any Swift modules from deps. Please file an issue on the Apple \
BUILD rules with a reproducible error case.
""")

        avoid_modules_list = sets.to_list(avoid_modules) if avoid_modules else None
        if avoid_modules_list:
            fail("""\
Error: Could not find a Swift module to build a Swift framework. This could be because "avoid_deps"\
 is too broadly defined.

Found these Swift modules within "avoid_deps": {avoid_modules_list}

Found these Swift modules within "deps": {transitive_modules_list}
""".format(
                avoid_modules_list = ", ".join(sorted(avoid_modules_list)),
                transitive_modules_list = ", ".join(sorted([
                    module.name
                    for module in transitive_modules_list
                ])),
            ))

        fail("""\
Internal Error: Could not find any Swift modules from deps, even though information for transitive \
modules from a Swift library dependency was found in deps. Please file an issue on the Apple BUILD \
rules with a reproducible error case.
""")

    return swift_module

def _modulemap_contents(
        *,
        framework_modulemap,
        module_name):
    """Returns the contents for the modulemap file for a Swift framework.

    Args:
        framework_modulemap: Boolean to indicate if the generated modulemap should be for a
            framework instead of a library or a generic module.
        module_name: The name of the Swift module.

    Returns:
        A string representing a generated modulemap.
    """
    return """\
{module_with_qualifier} {module_name} {{
  header "{module_name}.h"
  requires objc
}}
""".format(
        module_with_qualifier = "framework module" if framework_modulemap else "module",
        module_name = module_name,
    )

def _declare_modulemap(
        *,
        actions,
        framework_modulemap,
        label_name,
        module_name,
        output_discriminator):
    """Generates and declares the modulemap file for this Swift framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        framework_modulemap: Boolean to indicate if the generated modulemap should be for a
            framework instead of a library or a generic module.
        label_name: Name of the target being built.
        module_name: The name of the Swift module.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.

    Returns:
        A File referencing the intermediate generated modulemap.
    """
    modulemap = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "module.modulemap",
    )
    actions.write(modulemap, _modulemap_contents(
        framework_modulemap = framework_modulemap,
        module_name = module_name,
    ))
    return modulemap

def _declare_generated_header(
        *,
        actions,
        generated_header,
        label_name,
        module_name,
        output_discriminator):
    """Declares the generated header file for this Swift framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        generated_header: A File referencing the generated header from a SwiftInfo provider.
        label_name: Name of the target being built.
        module_name: The name of the Swift module.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.

    Returns:
        A File referencing the intermediate generated header.
    """
    bundle_header = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "{}.h".format(module_name),
    )
    actions.symlink(
        target_file = generated_header,
        output = bundle_header,
    )
    return bundle_header

def _declare_swiftdoc(
        *,
        actions,
        arch,
        label_name,
        output_discriminator,
        swiftdoc):
    """Declares the swiftdoc for this Swift framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        arch: The cpu architecture that the generated swiftdoc belongs to.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        swiftdoc: A File referencing the swiftdoc file from a SwiftInfo provider.

    Returns:
        A File referencing the intermediate swiftdoc.
    """
    bundle_doc = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "{}.swiftdoc".format(arch),
    )
    actions.symlink(
        target_file = swiftdoc,
        output = bundle_doc,
    )
    return bundle_doc

def _declare_swiftinterface(
        *,
        actions,
        arch,
        label_name,
        output_discriminator,
        swiftinterface):
    """Declares the swiftinterface for this Swift framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        arch: The cpu architecture that the generated swiftdoc belongs to.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        swiftinterface: A File referencing the swiftinterface file from a SwiftInfo provider.

    Returns:
        A File referencing the intermediate swiftinterface.
    """
    bundle_interface = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "{}.swiftinterface".format(arch),
    )
    actions.symlink(
        target_file = swiftinterface,
        output = bundle_interface,
    )
    return bundle_interface

def _declare_swiftmodule(
        *,
        actions,
        arch,
        label_name,
        output_discriminator,
        swiftmodule):
    """Declares the swiftmodule for this Swift framework.

    Args:
        actions: The actions provider from `ctx.actions`.
        arch: The cpu architecture that the generated swiftdoc belongs to.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        swiftmodule: A File referencing the swiftmodule file from a SwiftInfo provider.

    Returns:
        A File referencing the intermediate swiftmodule.
    """
    bundle_module = intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "{}.swiftmodule".format(arch),
    )
    actions.symlink(
        target_file = swiftmodule,
        output = bundle_module,
    )
    return bundle_module

swift_info_support = struct(
    verify_found_module_name = _verify_found_module_name,
    modules_from_avoid_deps = _modules_from_avoid_deps,
    swift_include_info = _swift_include_info,
    declare_modulemap = _declare_modulemap,
    declare_generated_header = _declare_generated_header,
    declare_swiftdoc = _declare_swiftdoc,
    declare_swiftinterface = _declare_swiftinterface,
    declare_swiftmodule = _declare_swiftmodule,
)
