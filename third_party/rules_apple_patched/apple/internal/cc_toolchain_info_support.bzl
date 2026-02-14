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

"""Support methods for handling CcToolchainInfo providers."""

def _get_apple_clang_triplet(cc_toolchain):
    """Parses and performs normalization on Clang target triplet string reference.

    The C++ ToolchainInfo provider `target_gnu_system_name` field references an LLVM target triple.
    This support method parses this target triplet and normalizes information for Apple targets.

    See: https://clang.llvm.org/docs/CrossCompilation.html#target-triple

    Args:
        cc_toolchain: CcToolchainInfo provider.
    Returns:
        A normalized Clang target triplet struct for Apple targets.
    """
    components = cc_toolchain.target_gnu_system_name.split("-")
    raw_triplet = struct(
        architecture = components[0],
        vendor = components[1],
        os = components[2],
        environment = components[3] if len(components) > 3 else None,
    )

    if raw_triplet.vendor != "apple":
        return raw_triplet

    environment = "device" if (raw_triplet.environment == None) else "simulator"

    # strip version from Apple platforms
    os = raw_triplet.os
    for index in range(len(raw_triplet.os)):
        if raw_triplet.os[index].isdigit():
            os = raw_triplet.os[:index]
            break

    # normalize MacOS names
    if os in ("macos", "macosx", "darwin"):
        os = "macos"

    return struct(
        architecture = raw_triplet.architecture,
        vendor = raw_triplet.vendor,
        os = os,
        environment = environment,
    )

cc_toolchain_info_support = struct(
    get_apple_clang_triplet = _get_apple_clang_triplet,
)
