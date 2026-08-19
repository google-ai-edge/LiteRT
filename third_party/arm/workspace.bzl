#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>.
# SPDX-License-Identifier: Apache-2.0
#

"""Workspace definitions for Arm(R) dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def arm_deps():
    http_archive(
        name = "arm_dep_vulkan_headers",
        build_file = "@//third_party/arm/vulkan-headers:BUILD.arm_dep_vulkan_headers",
        integrity = "sha256-17hHEvhGlle6o3pDbRoj778KY1T8iDW2dY7wNuFdzBQ=",
        strip_prefix = "Vulkan-Headers-1.4.349",
        urls = [
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.349.tar.gz",
        ],
    )

    http_archive(
        name = "arm_dep_spirv_headers",
        sha256 = "00284a33e1e19014723c8e88ca7a16e8988cd23f839ec2b7da6bb1808fd2a751",
        strip_prefix = "SPIRV-Headers-vulkan-sdk-1.4.328.0",
        urls = [
            "https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/vulkan-sdk-1.4.328.0.tar.gz",
        ],
    )

    http_archive(
        name = "ai_ml_sdk_vgf_library",
        build_file = "@//third_party/arm/ai-ml-sdk-vgf-library:BUILD.ai_ml_sdk_vgf_library",
        # This hard codes a flatbuffers version check that doesn't match
        # LiteRT's workspace dependency pin so a patch file is needed.
        patches = ["@//third_party/arm/ai-ml-sdk-vgf-library:PATCH.ai_ml_sdk_vgf_library"],
        sha256 = "15edaaae0107ca7588dcb8966915075e71099705119241c03be0f2cba6a9a654",
        strip_prefix = "ai-ml-sdk-vgf-library-0.9.0",
        urls = [
            "https://github.com/arm/ai-ml-sdk-vgf-library/archive/refs/tags/v0.9.0.tar.gz",
        ],
    )

    http_archive(
        name = "tosa_for_spirv_codegen",
        build_file = "@//third_party/arm/tosa-for-spirv-codegen:BUILD.tosa_for_spirv_codegen",
        sha256 = "7a222e03a38dfab4a09372a613a5be3895d33e5b92020fe6078e32e8d74fb0bf",
        strip_prefix = "tosa-for-spirv-codegen-2.0.0",
        urls = [
            "https://github.com/arm/tosa-for-spirv-codegen/archive/refs/tags/v2.0.0.tar.gz",
        ],
    )
