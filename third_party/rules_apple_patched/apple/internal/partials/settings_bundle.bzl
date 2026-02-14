# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Partial implementation for processing the settings bundle for iOS apps."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "//apple:providers.bzl",
    "AppleResourceInfo",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)
load(
    "//apple/internal:resources.bzl",
    "resources",
)
load(
    "//apple/internal/utils:bundle_paths.bzl",
    "bundle_paths",
)

def _settings_bundle_partial_impl(
        *,
        settings_bundle):
    """Implementation for the settings bundle processing partial."""

    if not settings_bundle:
        return struct()

    provider = settings_bundle[AppleResourceInfo]
    fields = resources.populated_resource_fields(provider)
    bundle_files = []
    for field in fields:
        for parent_dir, _, files in getattr(provider, field):
            bundle_name = bundle_paths.farthest_parent(parent_dir, "bundle")
            parent_dir = parent_dir.replace(bundle_name, "Settings.bundle")
            bundle_files.append((processor.location.resource, parent_dir, files))

    return struct(bundle_files = bundle_files)

def settings_bundle_partial(
        *,
        actions,
        platform_prerequisites,
        rule_label,
        settings_bundle):
    """Constructor for the settings bundles processing partial.

    This partial processes the settings bundle for Apple applications.

    Args:
        actions: The actions provider from `ctx.actions`.
        platform_prerequisites: Struct containing information on the platform being targeted.
        rule_label: The label of the target being analyzed.
        settings_bundle: A list of labels representing resource bundle targets that contain the
            files that make up the application's settings bundle.

    Returns:
        A partial that returns the bundle location of the settings bundle, if any were configured.
    """
    _unused = (actions, platform_prerequisites, rule_label)  # @unused

    return partial.make(
        _settings_bundle_partial_impl,
        settings_bundle = settings_bundle,
    )
