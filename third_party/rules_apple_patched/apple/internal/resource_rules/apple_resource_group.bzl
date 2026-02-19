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

"""Implementation of apple_resource_group rule."""

def _apple_resource_group_impl(_ctx):
    # All of the resource processing logic for this rule exists in the apple_resource_aspect.
    #
    # To transform the attributes referenced by this rule into resource providers, that aspect must
    # be used to iterate through all relevant instances of this rule in the build graph.
    return []

apple_resource_group = rule(
    implementation = _apple_resource_group_impl,
    attrs = {
        "resources": attr.label_list(
            allow_empty = True,
            allow_files = True,
            doc = """
Files to include in the final bundle that depends on this target. Files that are processable
resources, like .xib, .storyboard, .strings, .png, and others, will be processed by the Apple
bundling rules that have those files as dependencies. Other file types that are not processed will
be copied verbatim. These files are placed in the root of the final bundle (e.g.
Payload/foo.app/...) in most cases. However, if they appear to be localized (i.e. are contained in a
directory called *.lproj), they will be placed in a directory of the same name in the app bundle.

You can also add apple_resource_bundle and apple_bundle_import targets into `resources`, and the
resource bundle structures will be propagated into the final bundle.
""",
        ),
        "strip_structured_resources_prefixes": attr.string_list(
            doc = """
A list of prefixes to strip from the paths of structured resources. For each
structured resource, if the path starts with one of these prefixes, the first
matching prefix will be removed from the path when the resource is placed in
the bundle root. This is useful for removing intermediate directories from the
resource paths.

For example, if `structured_resources` contains `["intermediate/res/foo.png"]`,
and `strip_structured_resources_prefixes` contains `["intermediate"]`,
`res/foo.png` will end up inside the bundle.
""",
        ),
        "structured_resources": attr.label_list(
            allow_empty = True,
            allow_files = True,
            doc = """
Files to include in the final application bundle. They are not processed or compiled in any way
besides the processing done by the rules that actually generate them. These files are placed in the
bundle root in the same structure passed to this argument, so `["res/foo.png"]` will end up in
`res/foo.png` inside the bundle.
""",
        ),
    },
    doc = """
This rule encapsulates a target which provides resources to dependents. An
`apple_resource_group`'s `resources` and `structured_resources` are put in the
top-level Apple bundle target. `apple_resource_group` targets need to be added
to library targets through the `data` attribute, or to other
`apple_resource_bundle` or `apple_resource_group` targets through the
`resources` attribute.
""",
)
