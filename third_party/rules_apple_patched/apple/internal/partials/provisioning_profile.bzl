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

"""Partial implementation for embedding provisioning profiles."""

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

def _provisioning_profile_partial_impl(
        *,
        actions,
        extension,
        location,
        output_discriminator,
        profile_artifact,
        rule_label):
    """Implementation for the provisioning profile partial."""

    if not profile_artifact:
        fail(
            "\n".join([
                "ERROR: In {}:".format(str(rule_label)),
                "Building for device, but no provisioning_profile attribute was set.",
            ]),
        )

    # Create intermediate file with proper name for the binary.
    intermediate_file = intermediates.file(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = output_discriminator,
        file_name = "embedded.%s" % extension,
    )
    actions.symlink(
        target_file = profile_artifact,
        output = intermediate_file,
    )

    return struct(
        bundle_files = [
            (location, None, depset([intermediate_file])),
        ],
    )

def provisioning_profile_partial(
        *,
        actions,
        extension = "mobileprovision",
        location = processor.location.resource,
        output_discriminator = None,
        profile_artifact,
        rule_label):
    """Constructor for the provisioning profile partial.

    This partial propagates the bundle location for the embedded provisioning profile artifact for
    the target.

    Args:
      actions: The actions provider from `ctx.actions`.
      extension: The embedded provisioning profile extension.
      location: The location within the bundle to place the profile.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.
      profile_artifact: The provisioning profile to embed for this target.
      rule_label: The label of the target being analyzed.

    Returns:
      A partial that returns the bundle location of the provisioning profile artifact.
    """
    return partial.make(
        _provisioning_profile_partial_impl,
        actions = actions,
        extension = extension,
        location = location,
        output_discriminator = output_discriminator,
        profile_artifact = profile_artifact,
        rule_label = rule_label,
    )
