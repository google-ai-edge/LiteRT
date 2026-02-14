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

"""Implementation of the aspect that propagates framework providers."""

load(
    "//apple:providers.bzl",
    "AppleFrameworkImportInfo",
    "apple_provider",
)
load(
    "//apple/internal/providers:embeddable_info.bzl",
    "AppleEmbeddableInfo",
    "embeddable_info",
)

# List of attributes through which the aspect propagates. We include `runtime_deps` here as
# these are supported by `objc_library` for frameworks that should be present in the bundle, but not
# linked against.
# TODO(b/120205406): Remove `runtime_deps` support to use objc_library/swift_library `data` instead.
_FRAMEWORK_PROVIDERS_ASPECT_ATTRS = [
    "data",
    "deps",
    "frameworks",
    "implementation_deps",
    "private_deps",
    "runtime_deps",
]

def _framework_provider_aspect_impl(target, ctx):
    """Implementation of the framework provider propagation aspect."""
    if AppleFrameworkImportInfo in target and ctx.rule.kind != "objc_library":
        return []

    apple_framework_infos = []
    apple_embeddable_infos = []

    for attribute in _FRAMEWORK_PROVIDERS_ASPECT_ATTRS:
        if not hasattr(ctx.rule.attr, attribute):
            continue
        for dep_target in getattr(ctx.rule.attr, attribute):
            # *_framework_import targets support
            if AppleFrameworkImportInfo in dep_target:
                apple_framework_infos.append(dep_target[AppleFrameworkImportInfo])

            # *_framework targets support
            if AppleEmbeddableInfo in dep_target and ctx.rule.kind == "objc_library":
                apple_embeddable_infos.append(dep_target[AppleEmbeddableInfo])

    apple_framework_info = apple_provider.merge_apple_framework_import_info(apple_framework_infos)
    apple_embeddable_info = embeddable_info.merge_providers(apple_embeddable_infos)

    providers = []
    if apple_framework_info.framework_imports:
        providers.append(apple_framework_info)
    if apple_embeddable_info:
        providers.append(apple_embeddable_info)

    return providers

framework_provider_aspect = aspect(
    implementation = _framework_provider_aspect_impl,
    attr_aspects = _FRAMEWORK_PROVIDERS_ASPECT_ATTRS,
    doc = """
Aspect that collects transitive `AppleFrameworkImportInfo` providers from non-Apple rules targets
(e.g. `objc_library` or `swift_library`) to be packaged within the top-level application bundle.

Supported framework and XCFramework rules are:

*   `apple_dynamic_framework_import`
*   `apple_dynamic_xcframework_import`
*   `apple_static_framework_import`
*   `apple_static_xcframework_import`
""",
)
