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

"""Support methods for handling artifacts from CcInfo providers."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _get_all_deps(*, deps, split_deps_keys = []):
    """Returns a list of all dependencies from a Label list and optional split attribute keys.

    Args:
        deps: Label list of (split) dependencies to traverse.
        split_deps_keys: (optional) List of split attribute keys to use on split deps.

    Returns:
        List of all dependencies. If split_deps_keys is not provided, return deps.
    """
    if type(deps) == "list":
        return deps

    if not split_deps_keys:
        return deps.values()

    all_deps = []
    for split_deps_key in split_deps_keys:
        all_deps += deps[split_deps_key]
    return all_deps

def _get_sdk_frameworks(*, deps, split_deps_keys = [], include_weak = False):
    """Returns a depset of SDK frameworks linked to dependencies.

    Args:
        deps: Label list of (split) dependencies to traverse.
        split_deps_keys: (optional) List of split attribute keys to use on split deps.
        include_weak: include weak frameworks in the depset
    Returns
        Depset of SDK framework strings linked to dependencies.
    """
    all_deps = _get_all_deps(deps = deps, split_deps_keys = split_deps_keys)

    sdk_frameworks = []
    for dep in all_deps:
        if not CcInfo in dep:
            continue
        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            for index, user_link_flag in enumerate(linker_input.user_link_flags):
                if (user_link_flag == "-framework" or
                    (include_weak and user_link_flag == "-weak_framework")):
                    sdk_frameworks.append(linker_input.user_link_flags[index + 1])
                elif (user_link_flag.startswith("-Wl,-framework,") or
                      (include_weak and user_link_flag.startswith("-Wl,-weak_framework,"))):
                    sdk_frameworks.append(user_link_flag.split(",")[-1])

    return depset(sdk_frameworks)

def _get_sdk_dylibs(*, deps, split_deps_keys = []):
    """Returns a depset of SDK dylibs linked to dependencies.

    Args:
        deps: Label list of (split) dependencies to traverse.
        split_deps_keys: (optional) List of split attribute keys to use on split deps.
    Returns
        Depset of SDK framework strings linked to dependencies.
    """
    all_deps = _get_all_deps(deps = deps, split_deps_keys = split_deps_keys)

    sdk_dylibs = []
    for dep in all_deps:
        if not CcInfo in dep:
            continue

        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            for user_link_flag in linker_input.user_link_flags:
                if user_link_flag.startswith("-l"):
                    sdk_dylibs.append("lib" + user_link_flag[2:])

    return depset(sdk_dylibs)

cc_info_support = struct(
    get_sdk_dylibs = _get_sdk_dylibs,
    get_sdk_frameworks = _get_sdk_frameworks,
)
