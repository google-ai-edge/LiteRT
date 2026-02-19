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
    "//apple/internal:processor.bzl",
    "processor",
)

def _framework_headers_partial_impl(*, hdrs):
    """Implementation for the framework headers partial."""
    return struct(
        bundle_files = [
            (processor.location.bundle, "Headers", depset(hdrs)),
        ],
    )

def framework_headers_partial(*, hdrs):
    """Constructor for the framework headers partial.

    This partial bundles the headers for dynamic frameworks.

    Args:
      hdrs: The list of headers to bundle.

    Returns:
      A partial that returns the bundle location of the framework header artifacts.
    """
    return partial.make(
        _framework_headers_partial_impl,
        hdrs = hdrs,
    )
