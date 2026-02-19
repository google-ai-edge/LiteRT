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

"""Rule to create dummy AppleResourceInfo resources.

This rule is meant to be used only for rules_apple tests and are considered implementation details
that may change at any time. Please do not depend on this rule.
"""

load(
    "//apple/internal:providers.bzl",  # buildifier: disable=bzl-visibility
    "new_appleresourceinfo",
)

def _dummy_apple_resource_info_impl(ctx):
    output = ctx.actions.declare_file("{}.out".format(ctx.label.name))
    ctx.actions.write(output, "dummy file")
    outputs = depset([output])
    return [
        DefaultInfo(files = outputs),
        new_appleresourceinfo(
            unprocessed = [(None, None, outputs)],
            owners = depset(),
            unowned_resources = depset(),
        ),
    ]

dummy_apple_resource_info = rule(_dummy_apple_resource_info_impl)
