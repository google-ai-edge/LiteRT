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

"""Partial implementation for bundling header and modulemaps for external-facing frameworks."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)

def _get_link_declarations(dylibs = [], frameworks = []):
    """Returns the module map lines that link to the given dylibs and frameworks.

    Args:
      dylibs: A sequence of library names (which must begin with "lib") that will
          be referenced in the module map.
      frameworks: A sequence of framework names that will be referenced in the
          module map.

    Returns:
      A list of "link" and "link framework" lines that reference the given
      libraries and frameworks.
    """
    link_lines = []

    for dylib in dylibs:
        if not dylib.startswith("lib"):
            fail("Linked libraries must start with 'lib' but found %s" % dylib)
        link_lines.append('link "%s"' % dylib[3:])
    for framework in frameworks:
        link_lines.append('link framework "%s"' % framework)

    return link_lines

def _get_umbrella_header_declaration(basename):
    """Returns the module map line that references an umbrella header.

    Args:
      basename: The basename of the umbrella header file to be referenced in the
          module map.

    Returns:
      The module map line that references the umbrella header.
    """
    return 'umbrella header "%s"' % basename

def _create_modulemap(
        actions,
        framework_modulemap,
        output,
        module_name,
        sdk_dylibs,
        sdk_frameworks,
        umbrella_header_name):
    """Creates a modulemap for a framework.

    Args:
      actions: The actions module from a rule or aspect context.
      framework_modulemap: Boolean to indicate if the generated modulemap should be for a
          framework instead of a library or a generic module. Defaults to `True`.
      output: A declared `File` to which the module map will be written.
      module_name: The name of the module to declare in the module map file.
      sdk_dylibs: A list of system dylibs to list in the module.
      sdk_frameworks: A list of system frameworks to list in the module.
      umbrella_header_name: The basename of the umbrella header file, or None if
          there is no umbrella header.
    """
    declarations = []
    if umbrella_header_name:
        declarations.append(
            _get_umbrella_header_declaration(umbrella_header_name),
        )
    declarations.extend([
        "export *",
        "module * { export * }",
    ])
    declarations.extend(_get_link_declarations(sdk_dylibs, sdk_frameworks))

    content = (
        "{module_with_qualifier} {module_name} {{\n".format(
            module_with_qualifier = "framework module" if framework_modulemap else "module",
            module_name = module_name,
        ) +
        "\n".join(["  " + decl for decl in declarations]) +
        "\n}\n"
    )
    actions.write(output = output, content = content)

def _create_umbrella_header(actions, output, bundle_name, headers, framework_imports):
    """Creates an umbrella header that imports a list of other headers.

    Args:
      actions: The `actions` module from a rule or aspect context.
      output: A declared `File` to which the umbrella header will be written.
      bundle_name: The name of the output bundle.
      headers: A list of header files to be imported by the umbrella header.
      framework_imports: Whether to import with the framework style or directly quoted.
    """
    if framework_imports:
        import_lines = ["#import <%s/%s>" % (bundle_name, f.basename) for f in headers]
    else:
        import_lines = ['#import "%s"' % f.basename for f in headers]
    content = "\n".join(import_lines) + "\n"
    actions.write(output = output, content = content)

def _exported_headers(
        *,
        public_hdrs,
        generated_umbrella_header_file):
    """Determines if the generated umbrella header needs to be an output of public headers.

    Args:
      public_hdrs: The list of headers to bundle.
      generated_umbrella_header_file: The generated umbrella header file.
    """
    for public_hdr in public_hdrs:
        if public_hdr.basename == generated_umbrella_header_file.basename:
            return public_hdrs
    return public_hdrs + [generated_umbrella_header_file]

def _framework_header_modulemap_partial_impl(
        *,
        actions,
        bundle_name,
        framework_modulemap,
        hdrs,
        label_name,
        output_discriminator,
        sdk_dylibs,
        sdk_frameworks,
        umbrella_header):
    """Implementation for the sdk framework headers and modulemaps partial."""
    bundle_files = []

    umbrella_header_name = None
    if umbrella_header:
        umbrella_header_name = umbrella_header.basename
        bundle_files.append(
            (processor.location.bundle, "Headers", depset(hdrs + [umbrella_header])),
        )
    elif hdrs:
        umbrella_header_name = "{}.h".format(bundle_name)
        umbrella_header_file = intermediates.file(
            actions = actions,
            target_name = label_name,
            output_discriminator = output_discriminator,
            file_name = umbrella_header_name,
        )
        _create_umbrella_header(
            actions,
            umbrella_header_file,
            bundle_name,
            sorted(hdrs),
            framework_modulemap,
        )

        # Don't bundle the umbrella header if there is only one public header
        # which has the same name
        if len(hdrs) == 1 and hdrs[0].basename == umbrella_header_name:
            bundle_files.append(
                (processor.location.bundle, "Headers", depset(hdrs)),
            )
        else:
            exported_hdrs = _exported_headers(
                public_hdrs = hdrs,
                generated_umbrella_header_file = umbrella_header_file,
            )

            bundle_files.append(
                (processor.location.bundle, "Headers", depset(exported_hdrs)),
            )
    else:
        umbrella_header_name = None

    # Create a module map if there is a need for one (that is, if there are
    # headers or if there are dylibs/frameworks that the target depends on).
    if any([sdk_dylibs, sdk_frameworks, umbrella_header_name]):
        modulemap_file = intermediates.file(
            actions = actions,
            target_name = label_name + ".modulemaps",
            output_discriminator = output_discriminator,
            file_name = "module.modulemap",
        )
        _create_modulemap(
            actions = actions,
            framework_modulemap = framework_modulemap,
            output = modulemap_file,
            module_name = bundle_name,
            sdk_dylibs = sorted(sdk_dylibs.to_list() if sdk_dylibs else []),
            sdk_frameworks = sorted(sdk_frameworks.to_list() if sdk_frameworks else []),
            umbrella_header_name = umbrella_header_name,
        )
        bundle_files.append((processor.location.bundle, "Modules", depset([modulemap_file])))

    return struct(
        bundle_files = bundle_files,
    )

def framework_header_modulemap_partial(
        *,
        actions,
        bundle_name,
        framework_modulemap = True,
        hdrs,
        label_name,
        output_discriminator = None,
        sdk_dylibs = [],
        sdk_frameworks = [],
        umbrella_header = None):
    """Constructor for the framework headers and modulemaps partial.

    This partial bundles the headers and modulemaps for sdk frameworks.

    Args:
      actions: The actions provider from `ctx.actions`.
      bundle_name: The name of the output bundle.
      framework_modulemap: Boolean to indicate if the generated modulemap should be for a
          framework instead of a library or a generic module. Defaults to `True`.
      hdrs: The list of headers to bundle.
      label_name: Name of the target being built.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      sdk_dylibs: A list of dynamic libraries referenced by this framework.
      sdk_frameworks: A list of frameworks referenced by this framework.
      umbrella_header: An umbrella header to use instead of generating one. Optional.

    Returns:
      A partial that returns the bundle location of the sdk framework header and modulemap
      artifacts.
    """
    return partial.make(
        _framework_header_modulemap_partial_impl,
        actions = actions,
        bundle_name = bundle_name,
        framework_modulemap = framework_modulemap,
        hdrs = hdrs,
        label_name = label_name,
        output_discriminator = output_discriminator,
        sdk_dylibs = sdk_dylibs,
        sdk_frameworks = sdk_frameworks,
        umbrella_header = umbrella_header,
    )
