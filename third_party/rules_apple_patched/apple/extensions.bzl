# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Definitions for bzlmod module extensions."""

load("@bazel_features//:features.bzl", "bazel_features")
load("//apple:repositories.bzl", "apple_rules_dependencies")

def _non_module_deps_impl(module_ctx):
    apple_rules_dependencies(include_bzlmod_ready_dependencies = False)

    metadata_kwargs = {}
    if bazel_features.external_deps.extension_metadata_has_reproducible:
        metadata_kwargs["reproducible"] = True

    return module_ctx.extension_metadata(
        **metadata_kwargs
    )

non_module_deps = module_extension(implementation = _non_module_deps_impl)
