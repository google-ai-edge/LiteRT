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

"""Rule for creating header_maps."""

load("@build_bazel_rules_swift//swift:providers.bzl", "SwiftInfo")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_common")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

HeaderMapInfo = provider(
    doc = "Provides information about created `.hmap` (header map) files",
    fields = {
        "public_header_maps": "depset of paths of any public header maps",
    },
)

def _write_header_map(actions, header_map_tool, output, module_name, hdrs_lists):
    """Makes a binary hmap file by executing the header_map tool.

    Args:
        actions: a ctx.actions struct
        header_map_tool: an executable pointing to @bazel_build_rules_ios//rules/hmap:hmaptool
        output: the output file that will contain the built hmap
        module_name: the prefix to be used for header imports
        hdrs_lists: an array of enumerables containing headers to be added to the hmap
    """

    args = actions.args()
    if module_name:
        args.add("--module_name", module_name)

    args.add("--output", output)

    for hdrs in hdrs_lists:
        args.add_all(hdrs)

    args.set_param_file_format(format = "multiline")
    args.use_param_file("@%s")

    actions.run(
        mnemonic = "HmapWrite",
        arguments = [args],
        executable = header_map_tool,
        outputs = [output],
    )

def _header_map_impl(ctx):
    """Implementation of the header_map() rule.

    It creates a text file with mappings and creates an action that calls out to the header_map tool
    to convert it to a binary header_map file.
    """
    hdrs_lists = [ctx.files.hdrs] if ctx.files.hdrs else []

    for dep in ctx.attr.deps:
        found_headers = []
        if apple_common.Objc in dep:
            found_headers.append(getattr(dep[apple_common.Objc], "direct_headers", []))
        if CcInfo in dep:
            found_headers.append(dep[CcInfo].compilation_context.direct_headers)
        if not found_headers:
            fail("Direct header provider: '%s' listed in 'deps' does not have any direct headers to provide." % dep)
        hdrs_lists.extend(found_headers)

    hdrs_lists = [[h for h in hdrs if h.basename.endswith(".h")] for hdrs in hdrs_lists]

    _write_header_map(
        actions = ctx.actions,
        header_map_tool = ctx.executable._hmaptool,
        output = ctx.outputs.header_map,
        module_name = ctx.attr.module_name,
        hdrs_lists = hdrs_lists,
    )

    # TODO: use SwiftInfo directly when rules_apple sets min for rules_swift to v3+
    if hasattr(swift_common, "create_swift_info"):
        swift_info = swift_common.create_swift_info()
    else:
        swift_info = SwiftInfo()

    return [
        apple_common.new_objc_provider(),
        swift_info,
        CcInfo(
            compilation_context = cc_common.create_compilation_context(
                headers = depset([ctx.outputs.header_map]),
            ),
        ),
        HeaderMapInfo(
            public_header_maps = depset([ctx.outputs.header_map]),
        ),
    ]

header_map = rule(
    implementation = _header_map_impl,
    attrs = {
        "module_name": attr.string(
            mandatory = False,
            doc = "The prefix to be used for header imports",
        ),
        "hdrs": attr.label_list(
            mandatory = False,
            allow_files = True,
            doc = "The list of headers included in the header_map",
        ),
        "deps": attr.label_list(
            mandatory = False,
            providers = [[apple_common.Objc], [CcInfo]],
            doc = "Targets whose direct headers should be added to the list of hdrs and rooted at the module_name",
        ),
        "_hmaptool": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("//tools/hmaptool:hmaptool"),
        ),
    },
    outputs = {
        "header_map": "%{name}.hmap",
    },
    doc = """\
Creates a binary `.hmap` file from the given headers suitable for passing to clang.

Headermaps can be used in `-I` and `-iquote` compile flags (as well as in `includes`) to tell clang where to find headers.
This can be used to allow headers to be imported at a consistent path regardless of the package structure being used.

For example, if you have a package structure like this:

    ```
    MyLib/
        headers/
            MyLib.h
        MyLib.c
        BUILD
    ```

And you want to import `MyLib.h` from `MyLib.c` using angle bracket imports: `#import <MyLib/MyLib.h>`
You can create a header map like this:

    ```bzl
    header_map(
        name = "MyLib.hmap",
        hdrs = ["headers/MyLib.h"],
    )
    ```

This generates a binary headermap that looks like:

    ```
    MyLib.h -> headers/MyLib.h
    MyLib/MyLib.h -> headers/MyLib.h
    ```

Then update `deps`, `copts` and `includes` to use the header map:

    ```bzl
    objc_library(
        name = "MyLib",
        module_name = "MyLib",
        srcs = ["MyLib.c"],
        hdrs = ["headers/MyLib.h"],
        deps = [":MyLib.hmap"],
        copts = ["-I.]
        includes = ["MyLib.hmap"]
    )
    ```
    """,
)
