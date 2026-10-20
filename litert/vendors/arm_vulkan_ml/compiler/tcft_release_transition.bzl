# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Transitions used to build Arm ML extensions for Vulkan compiler dependencies."""

_ARM_VULKAN_ML_LLVM_ANDROID_COMPAT_COPTS = [
    "-UBACKTRACE_HEADER",
    "-UHAVE_BACKTRACE",
    "-UHAVE_POSIX_SPAWN",
    "-UHAVE_PTHREAD_GETNAME_NP",
]

def _arm_vulkan_ml_llvm_android_compat_copts(settings):
    for platform in settings["//command_line_option:platforms"]:
        if "android" in str(platform):
            return _ARM_VULKAN_ML_LLVM_ANDROID_COMPAT_COPTS
    return []

_TCFT_RELEASE_COPTS = [
    "-Oz",
    "-fdata-sections",
    "-ffunction-sections",
    "-fno-asynchronous-unwind-tables",
    "-fno-exceptions",
    "-fno-rtti",
    "-fno-unwind-tables",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-flto=full",
    "-fvirtual-function-elimination",
    "-fwhole-program-vtables",
]

def _arm_vulkan_ml_llvm_compat_transition_impl(settings, _attr):
    return {
        # Both compiler dependencies are shared libraries. Reuse PIC objects
        # instead of compiling a second set for their static archives.
        "//command_line_option:force_pic": True,
        "//command_line_option:copt": (
            settings["//command_line_option:copt"] +
            _arm_vulkan_ml_llvm_android_compat_copts(settings)
        ),
    }

_arm_vulkan_ml_llvm_compat_transition = transition(
    implementation = _arm_vulkan_ml_llvm_compat_transition_impl,
    inputs = [
        "//command_line_option:copt",
        "//command_line_option:platforms",
    ],
    outputs = [
        "//command_line_option:copt",
        "//command_line_option:force_pic",
    ],
)

def _tcft_release_transition_impl(settings, _attr):
    features = settings["//command_line_option:features"]
    if "-thin_lto" not in features:
        features = features + ["-thin_lto"]
    return {
        "//command_line_option:compilation_mode": "opt",
        "//command_line_option:force_pic": True,
        "//command_line_option:copt": (
            settings["//command_line_option:copt"] +
            _TCFT_RELEASE_COPTS +
            _arm_vulkan_ml_llvm_android_compat_copts(settings)
        ),
        "//command_line_option:features": features,
        "//command_line_option:linkopt": (
            settings["//command_line_option:linkopt"] +
            ["-flto=full"]
        ),
        "//command_line_option:strip": "always",
    }

_tcft_release_transition = transition(
    implementation = _tcft_release_transition_impl,
    inputs = [
        "//command_line_option:copt",
        "//command_line_option:features",
        "//command_line_option:linkopt",
        "//command_line_option:platforms",
    ],
    outputs = [
        "//command_line_option:compilation_mode",
        "//command_line_option:copt",
        "//command_line_option:features",
        "//command_line_option:force_pic",
        "//command_line_option:linkopt",
        "//command_line_option:strip",
    ],
)

def _tcft_release_file_impl(ctx):
    return [DefaultInfo(files = ctx.attr.src[0][DefaultInfo].files)]

arm_vulkan_ml_llvm_compat_file = rule(
    implementation = _tcft_release_file_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            cfg = _arm_vulkan_ml_llvm_compat_transition,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

tcft_release_file = rule(
    implementation = _tcft_release_file_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            cfg = _tcft_release_transition,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
