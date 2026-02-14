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

"""File references to important output files from the rule.

These file references can be used across the bundling logic, but there must be only 1 action
registered to generate these files.
"""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "//apple/internal:experimental.bzl",
    "is_experimental_tree_artifact_enabled",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)

def _archive(
        *,
        actions,
        bundle_extension,
        bundle_name,
        label_name,
        output_discriminator = "",
        platform_prerequisites,
        predeclared_outputs,
        rule_descriptor):
    """Returns a file reference for this target's archive."""
    bundle_name_with_extension = bundle_name + bundle_extension

    tree_artifact_enabled = is_experimental_tree_artifact_enabled(
        platform_prerequisites = platform_prerequisites,
    )
    if tree_artifact_enabled:
        if bundle_name != label_name:
            archive_relative_path = rule_descriptor.bundle_locations.archive_relative
            root_path = label_name + "_archive-root"
            return actions.declare_directory(
                paths.join(root_path, archive_relative_path, bundle_name_with_extension),
            )
        if output_discriminator:
            return actions.declare_directory(paths.join(
                output_discriminator,
                bundle_name_with_extension,
            ))

        return actions.declare_directory(bundle_name_with_extension)
    return predeclared_outputs.archive

def _archive_for_embedding(
        *,
        actions,
        bundle_name,
        bundle_extension,
        label_name,
        platform_prerequisites,
        predeclared_outputs,
        rule_descriptor):
    """Returns a files reference for this target's archive, when embedded in another target."""
    has_different_embedding_archive = _has_different_embedding_archive(
        platform_prerequisites = platform_prerequisites,
        rule_descriptor = rule_descriptor,
    )

    if has_different_embedding_archive:
        return actions.declare_file("%s.embedding.zip" % label_name)
    else:
        return _archive(
            actions = actions,
            bundle_extension = bundle_extension,
            bundle_name = bundle_name,
            label_name = label_name,
            platform_prerequisites = platform_prerequisites,
            predeclared_outputs = predeclared_outputs,
            rule_descriptor = rule_descriptor,
        )

def _binary(*, actions, bundle_name, executable_name, label_name, output_discriminator):
    """Returns a file reference for the binary that will be packaged into this target's archive. """
    file_name = executable_name if executable_name else bundle_name
    return intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = file_name,
    )

def _executable(*, actions, label_name):
    """Returns a file reference for the executable that would be invoked with `bazel run`."""
    return actions.declare_file(label_name)

def _dsyms(*, processor_result):
    """Returns a depset of all of the dsyms from the result."""
    dsyms = []
    for provider in processor_result.providers:
        if getattr(provider, "dsyms", None):
            dsyms.append(provider.dsyms)
    return depset(transitive = dsyms)

def _infoplist(*, actions, label_name, output_discriminator):
    """Returns a file reference for this target's Info.plist file."""
    return intermediates.file(
        actions = actions,
        target_name = label_name,
        output_discriminator = output_discriminator,
        file_name = "Info.plist",
    )

def _has_different_embedding_archive(*, platform_prerequisites, rule_descriptor):
    """Returns True if this target exposes a different archive when embedded in another target."""
    tree_artifact_enabled = is_experimental_tree_artifact_enabled(
        platform_prerequisites = platform_prerequisites,
    )
    if tree_artifact_enabled:
        return False
    return (
        rule_descriptor.bundle_locations.archive_relative != "" and
        rule_descriptor.expose_non_archive_relative_output
    )

def _merge_output_groups(*output_groups_list):
    """Merges a list of output group dictionaries into a single dictionary.

    In order to properly support use cases such as validation actions, which are represented as a
    uniquely named output group, partials or other parts of the build should propagate their output
    groups as raw dictionaries and collect them as lists of dictionaries and not wrap them into the
    `OutputGroupInfo` provider until the very end of the rule or aspect implementation function.

    In order to support this, this function merges these lists of output group dictionaries so that
    the final dictionary contains the keys from all of the dictionaries that were given to it, and
    furthermore, if two dictionaries in the list have the same key, the `depset`s for those keys
    will be merged.

    Args:
        *output_groups_list: A list of dictionaries that represent output groups; that is, their
            keys are strings (the names of the output groups) and their values are `depset`s of
            `File`s.

    Returns:
        A new dictionary containing the union of all of the output groups.
    """

    # We collect the depsets for each key as a flat list and create a single transitive depset
    # from all of them at the end, instead of creating a new chained depset each time we find a
    # duplicate key. This prevents the depset depth from growing too large (even though in
    # practice, this is unlikely for our uses of this function).
    output_group_depsets = {}
    for output_groups in output_groups_list:
        for name, files in output_groups.items():
            if name not in output_group_depsets:
                output_group_depsets[name] = [files]
            else:
                output_group_depsets[name].append(files)

    # We can unconditionally create a new depset here; if `depsets` has a length of 1, there is
    # a fast-path that returns the same depset instead of constructing a new nested one.
    merged_output_groups = {
        name: depset(transitive = depsets)
        for name, depsets in output_group_depsets.items()
    }

    return merged_output_groups

def _root_path_from_archive(*, archive):
    """Given an archive, returns a path to a directory reference for this target's archive root."""
    return paths.replace_extension(archive.path, "_archive-root")

outputs = struct(
    archive = _archive,
    archive_for_embedding = _archive_for_embedding,
    binary = _binary,
    dsyms = _dsyms,
    executable = _executable,
    infoplist = _infoplist,
    merge_output_groups = _merge_output_groups,
    root_path_from_archive = _root_path_from_archive,
    has_different_embedding_archive = _has_different_embedding_archive,
)
