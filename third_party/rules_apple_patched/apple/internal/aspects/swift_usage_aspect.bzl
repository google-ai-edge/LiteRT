# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""An aspect that collects information about Swift usage among dependencies."""

load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
)

SwiftUsageInfo = provider(
    doc = """\
A provider that indicates that Swift was used by a target or any target that it
depends on.
""",
    fields = {},
)

def _swift_usage_aspect_impl(target, aspect_ctx):
    # Targets can directly propagate their own `SwiftUsageInfo` provider. In
    # those case, this aspect must not propagate a `SwiftUsageInfo`, because
    # doing so results in a Bazel error.
    if SwiftUsageInfo in target:
        return []

    # If the target propagates `SwiftInfo`, then it or something in its
    # dependencies more than likely uses Swift.
    if SwiftInfo in target:
        return [SwiftUsageInfo()]

    # If one of the deps propagates `SwiftUsageInfo` provider, we can
    # repropagate that information. We currently make the assumption that all
    # Swift dependencies are built with the same toolchain (as in Bazel
    # toolchain, not Swift toolchain).
    for dep in getattr(aspect_ctx.rule.attr, "deps", []):
        if SwiftUsageInfo in dep:
            return [dep[SwiftUsageInfo]]

    # Don't propagate the provider at all if the target nor its dependencies use
    # Swift.
    return []

swift_usage_aspect = aspect(
    attr_aspects = ["deps"],
    doc = """\
Collects information about how Swift is used in a dependency tree.

When attached to an attribute, this aspect will propagate a `SwiftUsageInfo`
provider for any target found in that attribute that uses Swift, either directly
or deeper in its dependency tree. Conversely, if neither a target nor its
transitive dependencies use Swift, the `SwiftUsageInfo` provider will not be
propagated.

We use an aspect (as opposed to propagating this information through normal
providers returned by `swift_library`) because the information is needed if
Swift is used _anywhere_ in a dependency graph, even as dependencies of other
language rules that wouldn't know how to propagate the Swift-specific providers.
""",
    implementation = _swift_usage_aspect_impl,
)
