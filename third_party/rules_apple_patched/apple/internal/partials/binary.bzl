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

"""Partial implementation for binary processing for bundles."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "//apple/internal:outputs.bzl",
    "outputs",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)

def _binary_partial_impl(
        *,
        actions,
        binary_artifact,
        bundle_name,
        executable_name,
        label_name,
        output_discriminator):
    """Implementation for the binary processing partial."""

    # Create intermediate file with proper name for the binary.
    output_binary = outputs.binary(
        actions = actions,
        bundle_name = bundle_name,
        executable_name = executable_name,
        label_name = label_name,
        output_discriminator = output_discriminator,
    )
    actions.symlink(target_file = binary_artifact, output = output_binary)

    return struct(
        bundle_files = [
            (processor.location.binary, None, depset([output_binary])),
        ],
    )

def binary_partial(
        *,
        actions,
        binary_artifact,
        bundle_name,
        executable_name,
        label_name,
        output_discriminator = None):
    """Constructor for the binary processing partial.

    This partial propagates the bundle location for the main binary artifact for the target.

    Args:
      actions: The actions provider from ctx.actions.
      binary_artifact: The main binary artifact for this target.
      bundle_name: The name of the output bundle.
      executable_name: The name of the output executable.
      label_name: Name of the target being built.
      output_discriminator: A string to differentiate between different target intermediate files
          or `None`.

    Returns:
      A partial that returns the bundle location of the binary artifact.
    """
    return partial.make(
        _binary_partial_impl,
        actions = actions,
        binary_artifact = binary_artifact,
        bundle_name = bundle_name,
        executable_name = executable_name or bundle_name,
        label_name = label_name,
        output_discriminator = output_discriminator,
    )
