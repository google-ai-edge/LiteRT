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

"""Aspect implementation for Swift dynamic framework support."""

load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
)
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

SwiftDynamicFrameworkInfo = provider(
    fields = {
        "module_name": "The module name for the single swift_library dependency.",
        "swiftinterfaces": """
Dictionary of architecture to the generated swiftinterface file for that architecture.
""",
        "swiftdocs": """
Dictionary of architecture to the generated swiftdoc file for that architecture.
""",
        "generated_header": """
The generated Objective-C header for the single swift_library dependency.
""",
        "swiftmodules": """
Dictionary of architecture to the generated swiftmodule file for that architecture.
""",
        "modulemap": """
Generated modulemap for that architecture.
""",
    },
    doc = """
Provider that collects artifacts required to build a Swift-based dynamic framework.
""",
)

def _swift_target_for_dep(dep):
    """Returns the target for which the dependency was compiled.

    This is really hacky, but there's no easy way to acquire the Apple CPU for which the target was
    built. One option would be to let this aspect propagate transitively through deps and have
    another provider that propagates the CPU, but the model there gets a bit more complicated to
    follow. With this approach, we avoid propagating the aspect transitively as well.

    This should be cleaned up when b/141931700 is fixed (adding support for ctx.rule.split_attr).
    """
    for action in dep.actions:
        if action.mnemonic == "SwiftCompile":
            target_found = False
            for arg in action.argv:
                if target_found:
                    return arg
                if arg == "-target":
                    target_found = True
    return None

def _swift_arch_for_dep(dep):
    """Returns the architecture for which the dependency was built."""
    target = _swift_target_for_dep(dep)
    if not target:
        return None
    return target.split("-", 1)[0]

def _modulemap_contents(module_name):
    """Returns the contents for the modulemap file for the framework."""
    return """\
framework module {module_name} {{
  header "{module_name}.h"
  requires objc
}}
""".format(module_name = module_name)

def _swift_dynamic_framework_aspect_impl(target, ctx):
    """Aspect implementation for Swift dynamic framework support."""

    swiftdeps = [x for x in [target] if SwiftInfo in x]
    ccinfos = [x for x in [target] if CcInfo in x]

    # If there are no Swift dependencies, return nothing.
    if not swiftdeps:
        return []

    # Get the generated_header using the CcInfo provider
    generated_header = None
    for dep in ccinfos:
        headers = dep[CcInfo].compilation_context.headers.to_list()
        if headers:
            generated_header = headers.pop(0)

    # Collect all relevant artifacts for Swift dynamic framework generation.
    module_name = None
    swiftdocs = {}
    swiftmodules = {}
    modulemap_file = None
    for dep in swiftdeps:
        swiftinfo = dep[SwiftInfo]
        arch = _swift_arch_for_dep(dep)
        if not arch:
            return []

        swiftmodule = None
        swiftdoc = None
        for module in swiftinfo.transitive_modules.to_list():
            if not module.swift:
                continue
            module_name = module.name
            swiftmodule = module.swift.swiftmodule
            swiftdoc = module.swift.swiftdoc

        swiftdocs[arch] = swiftdoc
        swiftmodules[arch] = swiftmodule

        if generated_header:
            modulemap_file = ctx.actions.declare_file("{}_file.modulemap".format(module_name))
            ctx.actions.write(modulemap_file, _modulemap_contents(module_name))

    # Make sure that all dictionaries contain at least one module before returning the provider.
    if all([module_name, swiftdocs, swiftmodules]):
        return [
            SwiftDynamicFrameworkInfo(
                module_name = module_name,
                generated_header = generated_header,
                swiftdocs = swiftdocs,
                swiftmodules = swiftmodules,
                modulemap = modulemap_file,
            ),
        ]
    else:
        fail(
            """\
error: Could not find all required artifacts and information to build a Swift dynamic framework. \
Please file an issue with a reproducible error case.\
""",
        )

swift_dynamic_framework_aspect = aspect(
    implementation = _swift_dynamic_framework_aspect_impl,
    doc = """
Aspect that collects Swift information to construct a dynamic framework that supports Swift.
""",
)
