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

"""Temporary file to centralize configuration of the experimental bundling logic."""

load(
    "//apple/internal/utils:defines.bzl",
    "defines",
)

# TODO(b/266604130): Remove helper method once users of define flag have been migrated.
def is_experimental_tree_artifact_enabled(
        *,
        config_vars = None,
        platform_prerequisites = None):
    """Returns whether tree artifact outputs experiment is enabled.

    Args:
        config_vars: A reference to configuration variables, typically from `ctx.var`.
        platform_prerequisites: Struct containing information on the platform being targeted, if one
            exists for the rule.
    Returns:
        True if tree artifact outputs are enabled (via --define or build setting), False otherwise.
    """
    if not config_vars and not platform_prerequisites:
        fail("Internal error: should be called with either config_vars or platform_prerequisites")

    if platform_prerequisites and platform_prerequisites.build_settings.use_tree_artifacts_outputs:
        return True

    return defines.bool_value(
        config_vars = platform_prerequisites.config_vars if platform_prerequisites else config_vars,
        define_name = "apple.experimental.tree_artifact_outputs",
        default = False,
    )
