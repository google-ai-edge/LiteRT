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

"""Partial implementation for AppleDynamicFrameworkInfo configuration."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//apple/internal:providers.bzl", "new_appledynamicframeworkinfo")

def _framework_provider_partial_impl(
        *,
        actions,
        bin_root_path,
        binary_artifact,
        bundle_name,
        bundle_only,
        cc_features,
        cc_info,
        cc_toolchain,
        rule_label):
    """Implementation for the framework provider partial."""

    # Create a directory structure that the linker can use to reference this
    # framework. It follows the pattern of
    # any_path/MyFramework.framework/MyFramework. The absolute path and files are
    # propagated using the AppleDynamicFrameworkInfo provider.
    framework_dir = paths.join("frameworks", "%s.framework" % bundle_name)
    framework_file = actions.declare_file(
        paths.join(framework_dir, bundle_name),
    )
    actions.symlink(
        target_file = binary_artifact,
        output = framework_file,
    )

    absolute_framework_dir = paths.join(
        bin_root_path,
        rule_label.package,
        framework_dir,
    )

    library_to_link = cc_common.create_library_to_link(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = cc_features,
        dynamic_library = binary_artifact,
    )
    linker_input = cc_common.create_linker_input(
        owner = rule_label,
        libraries = depset([] if bundle_only else [library_to_link]),
    )
    wrapper_cc_info = cc_common.merge_cc_infos(
        cc_infos = [
            CcInfo(
                linking_context = cc_common.create_linking_context(
                    linker_inputs = depset(direct = [linker_input]),
                ),
            ),
            cc_info,
        ],
    )

    framework_provider = new_appledynamicframeworkinfo(
        binary = binary_artifact,
        cc_info = wrapper_cc_info,
        framework_dirs = depset([absolute_framework_dir]),
        framework_files = depset([framework_file]),
    )

    return struct(
        providers = [framework_provider],
    )

def framework_provider_partial(
        *,
        actions,
        bin_root_path,
        binary_artifact,
        bundle_name,
        bundle_only,
        cc_features,
        cc_info,
        cc_toolchain,
        rule_label):
    """Constructor for the framework provider partial.

    This partial propagates the AppleDynamicFrameworkInfo provider required by
    the linking step. It contains the necessary files and configuration so that
    the framework can be linked against. This is only required for dynamic
    framework bundles.

    Args:
      actions: The actions provider from `ctx.actions`.
      bin_root_path: The path to the root `-bin` directory.
      binary_artifact: The linked dynamic framework binary.
      bundle_name: The name of the output bundle.
      bundle_only: Only include the bundle but do not link the framework
      cc_features: List of enabled C++ features.
      cc_info: The CcInfo provider containing information about the
          targets linked into the dynamic framework.
      cc_toolchain: The C++ toolchain to use.
      rule_label: The label of the target being analyzed.

    Returns:
      A partial that returns the AppleDynamicFrameworkInfo provider used to link
      this framework into the final binary.

    """
    return partial.make(
        _framework_provider_partial_impl,
        actions = actions,
        bin_root_path = bin_root_path,
        binary_artifact = binary_artifact,
        bundle_name = bundle_name,
        bundle_only = bundle_only,
        cc_features = cc_features,
        cc_info = cc_info,
        cc_toolchain = cc_toolchain,
        rule_label = rule_label,
    )
