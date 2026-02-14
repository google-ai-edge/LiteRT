# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Rules related to Apple capabilities."""

load(
    "//apple:providers.bzl",
    "AppleBaseBundleIdInfo",
)
load(
    "//apple/internal:bundling_support.bzl",
    "bundling_support",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebasebundleidinfo",
    "new_applesharedcapabilityinfo",
)

visibility([
    "//apple/...",
    "//test/...",
])

def _apple_base_bundle_id_impl(ctx):
    """Implementation for the apple_base_bundle_id rule."""
    if ctx.attr.variant_name:
        base_bundle_id = ctx.attr.organization_id + "." + ctx.attr.variant_name
    else:
        base_bundle_id = ctx.attr.organization_id

    bundling_support.validate_bundle_id(base_bundle_id)

    return [new_applebasebundleidinfo(
        base_bundle_id = base_bundle_id,
    )]

apple_base_bundle_id = rule(
    _apple_base_bundle_id_impl,
    attrs = {
        "organization_id": attr.string(
            mandatory = True,
            doc = """
The organization ID that is expected to act as a prefix for a given set of bundles.
""",
        ),
        "variant_name": attr.string(
            mandatory = False,
            doc = """
The variant name, if any, to append between the organization ID and the evaluated bundle ID suffix.
Optional.
""",
        ),
    },
    doc = """
A rule to dictate the form that a given bundle rules's bundle ID prefix should take.

Use this rule to help standardize on a common prefix among a set of bundle rules, consistent with a
provisioning profile for code signing.
""",
)

def _apple_capability_set_impl(ctx):
    """Implementation for the apple_capability_set rule."""
    base_bundle_id = ""

    if ctx.attr.base_bundle_id:
        base_bundle_id = ctx.attr.base_bundle_id[AppleBaseBundleIdInfo].base_bundle_id

    return [new_applesharedcapabilityinfo(
        base_bundle_id = base_bundle_id,
    )]

apple_capability_set = rule(
    _apple_capability_set_impl,
    attrs = {
        "base_bundle_id": attr.label(
            mandatory = False,  # to support multiple capability sets.
            providers = [[AppleBaseBundleIdInfo]],
            doc = """
The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.
""",
        ),
    },
    doc = """
A rule to represent the capabilities that a code sign aware Apple bundle rule output should have.

Currently this is used to represent a shared base bundle ID. In the future, this will be extended to
allow for representing entitlements within rule definitions.
""",
)
