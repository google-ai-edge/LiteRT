# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""AppleEmbeddableInfo provider implementation for embeddable bundles propagation."""

_APPLE_EMBEDDABLE_INFO_FIELDS = {
    "app_clips": """
A depset with the zipped archives of bundles that need to be expanded into the
AppClips section of the packaging bundle.""",
    "extensions": """
A depset with the zipped archives of bundles that need to be expanded into the
Extensions section of the packaging bundle.""",
    "frameworks": """
A depset with the zipped archives of bundles that need to be expanded into the
Frameworks section of the packaging bundle.""",
    "plugins": """
A depset with the zipped archives of bundles that need to be expanded into the
PlugIns section of the packaging bundle.""",
    "signed_frameworks": """
A depset of strings referencing frameworks that have already been codesigned.""",
    "watch_bundles": """
A depset with the zipped archives of bundles that need to be expanded into the Watch section of
the packaging bundle. Only applicable for iOS applications.""",
    "xpc_services": """
A depset with the zipped archives of bundles that need to be expanded into the XPCServices section
of the packaging bundle. Only applicable for macOS applications.""",
}

AppleEmbeddableInfo = provider(
    doc = """
Internal provider used to propagate the different embeddable bundles that a
top-level bundling rule will need to package. For non Apple targets (e.g.
objc_library) this provider is propagated with by the embeddable_info_aspect.

Do not depend on this provider for non Apple rules.
""",
    fields = _APPLE_EMBEDDABLE_INFO_FIELDS,
)

def _merge_providers(apple_embeddable_infos):
    """Merges multiple `AppleEmbeddableInfo` providers into one.

    Merging multiple providers into one is needed to collect multiple
    framework providers from `framework_provider_aspect`.

    Args:
        apple_embeddable_infos: List of `AppleEmbeddableInfo` to be merged.
    Returns:
        Merged `AppleEmbeddableInfo` provider.
    """
    if not apple_embeddable_infos:
        return None

    embeddable_info_fields = {}
    for field in _APPLE_EMBEDDABLE_INFO_FIELDS:
        field_embeddable_infos = []
        for apple_embeddable_info in apple_embeddable_infos:
            if hasattr(apple_embeddable_info, field):
                field_embeddable_infos.append(
                    getattr(apple_embeddable_info, field),
                )

        if field_embeddable_infos:
            embeddable_info_fields[field] = depset(
                transitive = field_embeddable_infos,
            )

    return AppleEmbeddableInfo(**embeddable_info_fields)

embeddable_info = struct(
    merge_providers = _merge_providers,
)
