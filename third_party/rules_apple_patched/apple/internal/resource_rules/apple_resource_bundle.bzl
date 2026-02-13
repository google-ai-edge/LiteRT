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

"""Implementation of apple_resource_bundle rule."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple/internal:providers.bzl",
    "new_appleresourcebundleinfo",
)

def _apple_resource_bundle_impl(_ctx):
    # All of the resource processing logic for this rule exists in the apple_resource_aspect.
    #
    # To transform the attributes referenced by this rule into resource providers, that aspect must
    # be used to iterate through all relevant instances of this rule in the build graph.
    return [
        # TODO(b/122578556): Remove this ObjC provider instance.
        apple_common.new_objc_provider(),
        CcInfo(),
        new_appleresourcebundleinfo(),
    ]

apple_resource_bundle = rule(
    implementation = _apple_resource_bundle_impl,
    attrs = {
        "bundle_id": attr.string(
            doc = """
The bundle ID for this target. It will replace `$(PRODUCT_BUNDLE_IDENTIFIER)` found in the files
from defined in the `infoplists` paramter.
""",
        ),
        "bundle_name": attr.string(
            doc = """
The desired name of the bundle (without the `.bundle` extension). If this attribute is not set,
then the `name` of the target will be used instead.
""",
        ),
        "infoplists": attr.label_list(
            allow_empty = True,
            allow_files = True,
            doc = """
A list of `.plist` files that will be merged to form the `Info.plist` that represents the extension.
At least one file must be specified.
Please see [Info.plist Handling](/doc/common_info.md#infoplist-handling") for what is supported.

Duplicate keys between infoplist files
will cause an error if and only if the values conflict.
Bazel will perform variable substitution on the Info.plist file for the following values (if they
are strings in the top-level dict of the plist):

${BUNDLE_NAME}: This target's name and bundle suffix (.bundle or .app) in the form name.suffix.
${PRODUCT_NAME}: This target's name.
${TARGET_NAME}: This target's name.
The key in ${} may be suffixed with :rfc1034identifier (for example
${PRODUCT_NAME::rfc1034identifier}) in which case Bazel will replicate Xcode's behavior and replace
non-RFC1034-compliant characters with -.
""",
        ),
        "resources": attr.label_list(
            allow_empty = True,
            allow_files = True,
            doc = """
Files to include in the resource bundle. Files that are processable resources, like .xib,
.storyboard, .strings, .png, and others, will be processed by the Apple bundling rules that have
those files as dependencies. Other file types that are not processed will be copied verbatim. These
files are placed in the root of the resource bundle (e.g. `Payload/foo.app/bar.bundle/...`) in most
cases. However, if they appear to be localized (i.e. are contained in a directory called *.lproj),
they will be placed in a directory of the same name in the app bundle.

You can also add other `apple_resource_bundle` and `apple_bundle_import` targets into `resources`,
and the resource bundle structures will be propagated into the final bundle.
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
Files to include in the final resource bundle. They are not processed or compiled in any way
besides the processing done by the rules that actually generate them. These files are placed in the
bundle root in the same structure passed to this argument, so `["res/foo.png"]` will end up in
`res/foo.png` inside the bundle.
""",
        ),
    },
    doc = """
This rule encapsulates a target which is provided to dependers as a bundle. An
`apple_resource_bundle`'s resources are put in a resource bundle in the top
level Apple bundle dependent. apple_resource_bundle targets need to be added to
library targets through the `data` attribute.
""",
)
