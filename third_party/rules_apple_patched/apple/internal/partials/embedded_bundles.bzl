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

"""Partial implementation for processing embeddadable bundles."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "//apple/internal:experimental.bzl",
    "is_experimental_tree_artifact_enabled",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)
load(
    "//apple/internal/providers:embeddable_info.bzl",
    "AppleEmbeddableInfo",
)

def _embedded_bundles_partial_impl(
        *,
        bundle_embedded_bundles,
        embeddable_targets,
        platform_prerequisites,
        signed_frameworks,
        **input_bundles_by_type):
    """Implementation for the embedded bundles processing partial."""

    # Collect all _AppleEmbeddableInfo providers from the embeddable targets.
    embeddable_providers = [
        x[AppleEmbeddableInfo]
        for x in embeddable_targets
        if AppleEmbeddableInfo in x
    ]

    # Map of embedded bundle type to their final location in the top-level bundle.
    bundle_type_to_location = {
        "app_clips": processor.location.app_clip,
        "extensions": processor.location.extension,
        "frameworks": processor.location.framework,
        "plugins": processor.location.plugin,
        "watch_bundles": processor.location.watch,
        "xpc_services": processor.location.xpc_service,
    }

    transitive_bundles = dict()
    bundles_to_embed = []
    embeddedable_info_fields = {}

    tree_artifact_enabled = is_experimental_tree_artifact_enabled(
        platform_prerequisites = platform_prerequisites,
    )

    for bundle_type, bundle_location in bundle_type_to_location.items():
        for provider in embeddable_providers:
            if hasattr(provider, bundle_type):
                transitive_bundles.setdefault(
                    bundle_type,
                    [],
                ).append(getattr(provider, bundle_type))

        if bundle_embedded_bundles:
            # If this partial is configured to embed the transitive embeddable partials, collect
            # them into a list to be returned by this partial.
            if bundle_type in transitive_bundles:
                transitive_depset = depset(transitive = transitive_bundles.get(bundle_type, []))

                # With tree artifacts, we need to set the parent_dir of the file to be the basename
                # of the file. Expanding these depsets shouldn't be too much work as there shouldn't
                # be too many embedded targets per top-level bundle.
                if tree_artifact_enabled:
                    for bundle in transitive_depset.to_list():
                        bundles_to_embed.append(
                            (bundle_location, bundle.basename, depset([bundle])),
                        )
                else:
                    bundles_to_embed.append((bundle_location, None, transitive_depset))

            # Clear the transitive list of bundles for this bundle type since they will be packaged
            # in the bundle processing this partial and do not need to be propagated.
            transitive_bundles[bundle_type] = []

        # Construct the AppleEmbeddableInfo provider field for the bundle type being processed.
        # At this step, we inject the bundles that are inputs to this partial, since that propagates
        # the info for a higher level bundle to embed this bundle.
        if input_bundles_by_type.get(bundle_type) or transitive_bundles.get(bundle_type):
            embeddedable_info_fields[bundle_type] = depset(
                input_bundles_by_type.get(bundle_type, []),
                transitive = transitive_bundles.get(bundle_type, []),
            )

    # Construct the output files fields. If tree artifacts is enabled, propagate the bundles to
    # package into bundle_files. Otherwise, propagate through bundle_zips so that they can be
    # extracted.
    partial_output_fields = {}
    if tree_artifact_enabled:
        partial_output_fields["bundle_files"] = bundles_to_embed
    else:
        partial_output_fields["bundle_zips"] = bundles_to_embed

    # Construct a transitive depset of signed paths, indicating files that we expect to have
    # already been code signed by past targets.
    transitive_signed_framework_depsets = []

    # See if any signed_frameworks have been propagated.
    for provider in embeddable_providers:
        if hasattr(provider, "signed_frameworks"):
            transitive_signed_framework_depsets.append(provider.signed_frameworks)

    if transitive_signed_framework_depsets:
        # Output the existing, propagated signed_frameworks as an output of this partial.
        #
        # NOTE: We avoid passing this target's additional depset of signed_frameworks to the code
        # signing phase to avoid suggesting to this target's code signing phase that files that
        # will be code signed in this target have already been code signed.
        partial_output_fields["signed_frameworks"] = depset(
            transitive = transitive_signed_framework_depsets,
        )

        # Propagate the full set of signed frameworks upstream as the provider output.
        embeddedable_info_fields["signed_frameworks"] = depset(
            transitive = [signed_frameworks] + transitive_signed_framework_depsets,
        )
    else:
        # If no transitive signed frameworks were found, pass signed_frameworks.
        embeddedable_info_fields["signed_frameworks"] = signed_frameworks

    return struct(
        providers = [AppleEmbeddableInfo(**embeddedable_info_fields)],
        **partial_output_fields
    )

def embedded_bundles_partial(
        *,
        app_clips = [],
        bundle_embedded_bundles = False,
        embeddable_targets = [],
        extensions = [],
        frameworks = [],
        platform_prerequisites,
        plugins = [],
        signed_frameworks = depset(),
        watch_bundles = [],
        xpc_services = []):
    """Constructor for the embedded bundles processing partial.

    This partial is used to propagate and package embedded bundles into their respective locations
    inside top level bundling targets. Embeddable bundles are considered to be extensions (i.e.
    ExtensionKit extensions), frameworks, plugins (i.e. NSExtension extensions) and watchOS
    applications in the case of ios_application.

    Args:
        app_clips: List of plugin bundles that should be propagated downstream for a top level
            target to bundle inside `AppClips`.
        bundle_embedded_bundles: If True, this target will embed all transitive embeddable_bundles
            _only_ propagated through the targets given in embeddable_targets. If False, the
            embeddable bundles will be propagated downstream for a top level target to bundle them.
        embeddable_targets: The list of targets that propagate embeddable bundles to bundle or
            propagate.
        extensions: List of ExtensionKit extension bundles that should be propagated downstream for
            a top level target to bundle inside `Extensions`.
        frameworks: List of framework bundles that should be propagated downstream for a top level
            target to bundle inside `Frameworks`.
        platform_prerequisites: Struct containing information on the platform being targeted.
        plugins: List of plugin bundles that should be propagated downstream for a top level
            target to bundle inside `PlugIns`.
        signed_frameworks: A depset of strings referencing frameworks that have already been
            codesigned.
        watch_bundles: List of watchOS application bundles that should be propagated downstream for
            a top level target to bundle inside `Watch`.
        xpc_services: List of macOS XPC Service bundles that should be propagated downstream for
            a top level target to bundle inside `XPCServices`.

    Returns:
          A partial that propagates and/or packages embeddable bundles.
    """
    return partial.make(
        _embedded_bundles_partial_impl,
        app_clips = app_clips,
        bundle_embedded_bundles = bundle_embedded_bundles,
        embeddable_targets = embeddable_targets,
        extensions = extensions,
        frameworks = frameworks,
        platform_prerequisites = platform_prerequisites,
        plugins = plugins,
        signed_frameworks = signed_frameworks,
        watch_bundles = watch_bundles,
        xpc_services = xpc_services,
    )
