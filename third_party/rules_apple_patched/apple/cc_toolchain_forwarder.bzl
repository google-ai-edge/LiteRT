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

"""
A rule for handling the cc_toolchains and their constraints for a potential "fat" Mach-O binary.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("//apple:providers.bzl", "ApplePlatformInfo")
load("//apple/internal:providers.bzl", "new_appleplatforminfo")

def _target_os_from_rule_ctx(ctx):
    """Returns a `String` representing the selected Apple OS."""
    ios_constraint = ctx.attr._ios_constraint[platform_common.ConstraintValueInfo]
    macos_constraint = ctx.attr._macos_constraint[platform_common.ConstraintValueInfo]
    tvos_constraint = ctx.attr._tvos_constraint[platform_common.ConstraintValueInfo]
    visionos_constraint = ctx.attr._visionos_constraint[platform_common.ConstraintValueInfo]
    watchos_constraint = ctx.attr._watchos_constraint[platform_common.ConstraintValueInfo]

    if ctx.target_platform_has_constraint(ios_constraint):
        return str(apple_common.platform_type.ios)
    elif ctx.target_platform_has_constraint(macos_constraint):
        return str(apple_common.platform_type.macos)
    elif ctx.target_platform_has_constraint(tvos_constraint):
        return str(apple_common.platform_type.tvos)
    elif ctx.target_platform_has_constraint(visionos_constraint):
        return str(getattr(apple_common.platform_type, "visionos", None))
    elif ctx.target_platform_has_constraint(watchos_constraint):
        return str(apple_common.platform_type.watchos)
    fail("ERROR: A valid Apple platform constraint could not be found from the resolved toolchain.")

def _target_arch_from_rule_ctx(ctx):
    """Returns a `String` representing the selected target architecture or cpu type."""
    arm64_constraint = ctx.attr._arm64_constraint[platform_common.ConstraintValueInfo]
    arm64e_constraint = ctx.attr._arm64e_constraint[platform_common.ConstraintValueInfo]
    arm64_32_constraint = ctx.attr._arm64_32_constraint[platform_common.ConstraintValueInfo]
    armv7k_constraint = ctx.attr._armv7k_constraint[platform_common.ConstraintValueInfo]
    x86_64_constraint = ctx.attr._x86_64_constraint[platform_common.ConstraintValueInfo]

    if ctx.target_platform_has_constraint(arm64_constraint):
        return "arm64"
    elif ctx.target_platform_has_constraint(arm64e_constraint):
        return "arm64e"
    elif ctx.target_platform_has_constraint(arm64_32_constraint):
        return "arm64_32"
    elif ctx.target_platform_has_constraint(armv7k_constraint):
        return "armv7k"
    elif ctx.target_platform_has_constraint(x86_64_constraint):
        return "x86_64"
    fail("ERROR: A valid Apple cpu constraint could not be found from the resolved toolchain.")

def _target_environment_from_rule_ctx(ctx):
    """Returns a `String` representing the selected environment (e.g. "device", "simulator")."""
    simulator_constraint = ctx.attr._apple_simulator_constraint[platform_common.ConstraintValueInfo]
    if ctx.target_platform_has_constraint(simulator_constraint):
        return "simulator"

    return "device"

def _cc_toolchain_forwarder_impl(ctx):
    return [
        find_cpp_toolchain(ctx),
        new_appleplatforminfo(
            target_os = _target_os_from_rule_ctx(ctx),
            target_arch = _target_arch_from_rule_ctx(ctx),
            target_environment = _target_environment_from_rule_ctx(ctx),
        ),
    ]

cc_toolchain_forwarder = rule(
    implementation = _cc_toolchain_forwarder_impl,
    attrs = {
        # Legacy style toolchain assignment.
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        # The full list of possible constraints to be resolved are assigned here.
        "_ios_constraint": attr.label(
            default = Label("@platforms//os:ios"),
        ),
        "_macos_constraint": attr.label(
            default = Label("@platforms//os:macos"),
        ),
        "_tvos_constraint": attr.label(
            default = Label("@platforms//os:tvos"),
        ),
        "_visionos_constraint": attr.label(
            default = Label("@platforms//os:visionos"),
        ),
        "_watchos_constraint": attr.label(
            default = Label("@platforms//os:watchos"),
        ),
        "_arm64_constraint": attr.label(
            default = Label("@platforms//cpu:arm64"),
        ),
        "_arm64e_constraint": attr.label(
            default = Label("@platforms//cpu:arm64e"),
        ),
        "_arm64_32_constraint": attr.label(
            default = Label("@platforms//cpu:arm64_32"),
        ),
        "_armv7k_constraint": attr.label(
            default = Label("@platforms//cpu:armv7k"),
        ),
        "_x86_64_constraint": attr.label(
            default = Label("@platforms//cpu:x86_64"),
        ),
        "_apple_device_constraint": attr.label(
            default = Label("@build_bazel_apple_support//constraints:device"),
        ),
        "_apple_simulator_constraint": attr.label(
            default = Label("@build_bazel_apple_support//constraints:simulator"),
        ),
    },
    doc = """
Shared rule that returns CcToolchainInfo, plus a rules_apple defined provider based on querying
ctx.target_platform_has_constraint(...) that covers all Apple cpu, platform, environment constraints
for the purposes of understanding what constraints the results of each Apple split transition
resolve to from the perspective of any bundling and binary rules that generate "fat" Apple binaries.
""",
    provides = [cc_common.CcToolchainInfo, ApplePlatformInfo],
    # Anticipated "new" toolchain assignment.
    toolchains = use_cpp_toolchain(),
)
