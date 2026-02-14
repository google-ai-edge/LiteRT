# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Support macros to assist in detecting build features."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")

def _compute_enabled_features(*, requested_features, unsupported_features):
    """Returns a list of features for the given build.

    Args:
      requested_features: A list of features requested. Typically from `ctx.features`.
      unsupported_features: A list of features to ignore. Typically from `ctx.disabled_features`.

    Returns:
      A list containing the subset of features that should be used.
    """
    enabled_features_set = sets.make(requested_features)
    enabled_features_set = sets.difference(
        enabled_features_set,
        sets.make(unsupported_features),
    )
    return sets.to_list(enabled_features_set)

features_support = struct(
    compute_enabled_features = _compute_enabled_features,
)
