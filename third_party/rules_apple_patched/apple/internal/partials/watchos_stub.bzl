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

"""Partial implementation for placing the watchOS stub file in the archive."""

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

_AppleWatchosStubInfo = provider(
    doc = """
Private provider to propagate the watchOS stub that needs to be package in the iOS archive.
""",
    fields = {
        "binary": """
File artifact that contains a reference to the stub binary that needs to be packaged in the iOS
archive.
""",
    },
)

def _watchos_stub_partial_impl(
        *,
        actions,
        binary_artifact,
        label_name,
        output_discriminator,
        watch_application):
    """Implementation for the watchOS stub processing partial."""

    bundle_files = []
    providers = []
    if binary_artifact:
        # Create intermediate file with proper name for the binary.
        intermediate_file = intermediates.file(
            actions = actions,
            target_name = label_name,
            output_discriminator = output_discriminator,
            file_name = "WK",
        )
        actions.symlink(
            target_file = binary_artifact,
            output = intermediate_file,
        )
        bundle_files.append(
            (processor.location.bundle, "_WatchKitStub", depset([intermediate_file])),
        )
        providers.append(_AppleWatchosStubInfo(binary = intermediate_file))

    if watch_application:
        binary_artifact = watch_application[_AppleWatchosStubInfo].binary
        bundle_files.append(
            (processor.location.archive, "WatchKitSupport2", depset([binary_artifact])),
        )

    return struct(
        bundle_files = bundle_files,
        providers = providers,
    )

def watchos_stub_partial(
        *,
        actions,
        binary_artifact = None,
        label_name,
        output_discriminator = None,
        watch_application = None):
    """Constructor for the watchOS stub processing partial.

    This partial copies the WatchKit stub into the expected location inside the watchOS bundle.
    This partial only applies to the watchos_application rule for bundling the WK stub binary, and
    to the ios_application rule for packaging the stub in the WatchKitSupport2 root directory.

    Args:
        actions: The actions provider from `ctx.actions`.
        binary_artifact: The stub binary to copy.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        watch_application: Optional. A reference to the watch application associated with the rule.
            If defined, this partial will package the watchOS stub binary in the archive root.

    Returns:
      A partial that returns the bundle location of the stub binary.
    """
    return partial.make(
        _watchos_stub_partial_impl,
        actions = actions,
        binary_artifact = binary_artifact,
        label_name = label_name,
        output_discriminator = output_discriminator,
        watch_application = watch_application,
    )
