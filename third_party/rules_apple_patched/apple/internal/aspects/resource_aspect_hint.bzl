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

"""Implementation of the `apple_resource_hint` rule."""

# Possible actions for the AppleResourceHintInfo are
#     `resources` collect all labels referenced in the data attribute, process them based on
#                 file extension, flatten the folder heirarchy and include them in Contents/Resources
#     `runfiles`  collect all runfiles without processing and include them in Contents/Resources.
#     `suppress`  stop any collection of resources on this target. Transitive runfiles may still be
#                 collected based on ancestor resource rules.
apple_resource_hint_action = struct(
    resources = "resources",
    runfiles = "runfiles",
    suppress = "suppress",
)

_resource_actions = [
    apple_resource_hint_action.resources,
    apple_resource_hint_action.runfiles,
    apple_resource_hint_action.suppress,
]

AppleResourceHintInfo = provider(
    doc = """
Provider that propagates desire to automatically bundle runfiles, resources, or suppress both. 
Available actions are
    `resources` collect all labels referenced in the data attribute, process them based on
                file extension, flatten the folder heirarchy and include them in Contents/Resources
    `runfiles`  collect all runfiles without processing and include them in Contents/Resources.
    `suppress`  stop any collection of resources on this target. Transitive runfiles may still be
                collected based on ancestor resource rules.
""",
    fields = {
        "action": "Which resource action to run.",
    },
)

def _apple_resource_hint_impl(ctx):
    if ctx.attr.action not in _resource_actions:
        fail(str(ctx.label) + " apple resource hint allowed to take values: (" +
             ", ".join(_resource_actions) + ") but was set to unallowed value: " +
             ctx.attr.action)
    return AppleResourceHintInfo(
        action = ctx.attr.action,
    )

apple_resource_hint = rule(
    attrs = {
        "action": attr.string(
            mandatory = True,
            doc = """
Hints the resource collector to take a specific action.
Available actions are
    `resources` collect all labels referenced in the data attribute, process them based on
                file extension, flatten the folder heirarchy and include them in Contents/Resources
    `runfiles`  collect all runfiles without processing and include them in Contents/Resources.
    `suppress`  stop any collection of resources on this target. Transitive runfiles may still be
                collected based on ancestor resource rules.
""",
        ),
    },
    doc = """\
Defines an aspect hint that generates an appropriate AppleResourceHintInfo based on the
runfiles for this target.

> [!NOTE]
> Bazel 6 users must set the `--experimental_enable_aspect_hints` flag to utilize
> this rule. In addition, downstream consumers of rules that utilize this rule
> must also set the flag. The flag is enabled by default in Bazel 7.

Some rules like `cc_library` may have data associated with them in the data attribute
that is needed at runtime. If the library was linked in a `cc_binary` then those data
files would be made available to the application as `runfiles`. To control this
functionality with a `macos_application` you may use this aspect hint.


#### Collect resources of a cc_library

By default a cc_library will add its runfiles to the Contents/Resources folder of a
`macos_application`. To alter this behavior and have it collect resources instead
you can add this pre-built aspect hint. This will cause resources to be collected
and processed like objc_library.

```build
# //my/project/BUILD
cc_library(
    name = "somelib",
    data = ["mydata.txt"],
    aspect_hints = ["//apple:use_resources"],
)
```

#### Collect runfiles of a objc_library

Similar to above, you can modify the default resource collection behavior of
an objc_library by adding an aspect hint to `use_runfiles` instead of resources.

```build
# //my/project/BUILD
objc_library(
    name = "somelib",
    data = ["mydata.txt"],
    aspect_hints = ["//apple:use_runfiles"],
)
```

Runfiles are described here: https://bazel.build/contribute/codebase#runfiles
The runfiles tree reflects the project tree layout. For example, if you have these
files in your project
```
myFile1.txt
some_folder/myFile2.txt
```
with a build file
```build
cc_library(
    name = "somelib",
    data = ["myFile1.txt", "some_folder/myFile2.txt"],
    aspect_hints = ["//apple:use_runfiles"],
)
```
then the resources will be in
```
Contents/Resources/myFile1.txt
Contents/Resources/some_folder/myFile2.txt
```

#### Suppress resource collection

In some situations you may wish to suppress resource or runfile collection of
a target. You can add the `suppress_resources` aspect hint to accomplish this.
Note that runfile collection is transitive, so if an ancestor of this target
collects runfiles then this targets runfiles will be included regardless of
any aspect hint applied.

```build
# //my/project/BUILD
objc_library(
    name = "somelib",
    data = ["mydata.txt"],
    aspect_hints = ["//apple:suppress_resources"],
)
```
""",
    implementation = _apple_resource_hint_impl,
)
