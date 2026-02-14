# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""# Bazel rules for working with dtrace."""

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
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple/internal/utils:bundle_paths.bzl",
    "bundle_paths",
)

def _dtrace_compile_impl(ctx):
    """Implementation for dtrace_compile."""
    apple_fragment = ctx.fragments.apple
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    output_hdrs = []
    include_dir = None

    dtrace = "/usr/sbin/dtrace"
    if ctx.executable.dtrace:
        dtrace = ctx.executable.dtrace

    for src in ctx.files.srcs:
        owner_relative_path = bundle_paths.owner_relative_path(src)
        label_scoped_owner_path = ctx.label.name + "/" + owner_relative_path.lstrip("/")
        hdr = ctx.actions.declare_file(
            paths.replace_extension(label_scoped_owner_path, ".h"),
        )
        output_hdrs.append(hdr)
        apple_support.run(
            actions = ctx.actions,
            xcode_config = xcode_config,
            apple_fragment = apple_fragment,
            inputs = [src],
            outputs = [hdr],
            mnemonic = "dtraceCompile",
            executable = dtrace,
            arguments = ["-h", "-s", src.path, "-o", hdr.path],
            progress_message = ("Compiling dtrace probes %s" % (src.basename)),
        )

        if not include_dir:
            hdr_suffix = paths.replace_extension("/" + owner_relative_path.lstrip("/"), ".h")
            include_dir = hdr.path.removesuffix(hdr_suffix)

    return [
        apple_common.new_objc_provider(
            strict_include = depset([include_dir]),
        ),
        CcInfo(
            compilation_context = cc_common.create_compilation_context(
                headers = depset(output_hdrs),
            ),
        ),
        DefaultInfo(files = depset(output_hdrs)),
    ]

dtrace_compile = rule(
    implementation = _dtrace_compile_impl,
    attrs = dicts.add(apple_support.action_required_attrs(), {
        "dtrace": attr.label(
            doc = "dtrace binary to use.",
            mandatory = False,
            executable = True,
            cfg = "exec",
        ),
        "srcs": attr.label_list(
            allow_files = [".d"],
            allow_empty = False,
            doc = "dtrace(.d) source files to be compiled.",
        ),
    }),
    exec_compatible_with = [
        "@platforms//os:macos",
    ],
    fragments = ["apple"],
    doc = """
Compiles
[dtrace files with probes](https://www.ibm.com/developerworks/aix/library/au-dtraceprobes.html)
to generate header files to use those probes in C languages. The header files
generated will have the same name as the source files but with a `.h`
extension. Headers will be generated in a label scoped workspace relative file
structure. For example with a directory structure of

```
  Workspace
  foo/
    bar.d
```
and a target named `dtrace_gen` the header path would be
`<GENFILES>/dtrace_gen/foo/bar.h`.
""",
)
