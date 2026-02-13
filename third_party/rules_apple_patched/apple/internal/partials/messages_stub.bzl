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

"""Partial implementation for placing the messages support stub file in the archive."""

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

_AppleMessagesStubInfo = provider(
    doc = """
Private provider to propagate the messages stub that needs to be package in the iOS archive.
""",
    fields = {
        "binary": """
File artifact that contains a reference to the stub binary that needs to be packaged in the iOS
archive.
""",
    },
)

def _messages_stub_partial_impl(
        *,
        actions,
        binary_artifact,
        extensions,
        label_name,
        output_discriminator,
        package_messages_support):
    """Implementation for the messages support stub processing partial."""

    bundle_files = []
    providers = []

    if package_messages_support:
        extension_binaries = [
            x[_AppleMessagesStubInfo].binary
            for x in extensions
            if _AppleMessagesStubInfo in x
        ]

        if extension_binaries:
            bundle_files.append(
                (
                    processor.location.archive,
                    "MessagesApplicationExtensionSupport",
                    depset([extension_binaries[0]]),
                ),
            )

        if binary_artifact:
            intermediate_file = intermediates.file(
                actions = actions,
                target_name = label_name,
                output_discriminator = output_discriminator,
                file_name = "MessagesApplicationSupportStub",
            )
            actions.symlink(
                target_file = binary_artifact,
                output = intermediate_file,
            )

            bundle_files.append(
                (
                    processor.location.archive,
                    "MessagesApplicationSupport",
                    depset([intermediate_file]),
                ),
            )

    elif binary_artifact:
        intermediate_file = intermediates.file(
            actions = actions,
            target_name = label_name,
            output_discriminator = output_discriminator,
            file_name = "MessagesApplicationExtensionSupportStub",
        )
        actions.symlink(
            target_file = binary_artifact,
            output = intermediate_file,
        )
        providers.append(_AppleMessagesStubInfo(binary = intermediate_file))

    return struct(
        bundle_files = bundle_files,
        providers = providers,
    )

def messages_stub_partial(
        *,
        actions,
        binary_artifact = None,
        extensions = None,
        label_name,
        output_discriminator = None,
        package_messages_support = False):
    """Constructor for the messages support stub processing partial.

    This partial copies the messages support stubs into the expected location for iOS archives.

    Args:
        actions: The actions provider from `ctx.actions`.
        binary_artifact: The stub binary to copy.
        extensions: List of extension bundles associated with this rule.
        label_name: Name of the target being built.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        package_messages_support: Whether to package the messages stub binary in the archive root.

    Returns:
        A partial that returns the bundle location of the stub binaries.
    """
    return partial.make(
        _messages_stub_partial_impl,
        actions = actions,
        binary_artifact = binary_artifact,
        extensions = extensions,
        label_name = label_name,
        output_discriminator = output_discriminator,
        package_messages_support = package_messages_support,
    )
