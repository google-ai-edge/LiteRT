#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>.
# SPDX-License-Identifier: Apache-2.0
#

"""Workspace definitions for Arm ML extensions for Vulkan dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def arm_vulkan_ml_deps():
    http_archive(
        name = "arm_vulkan_ml_dep_vulkan_headers",
        build_file = "@//third_party/arm_vulkan_ml/vulkan-headers:BUILD.arm_vulkan_ml_dep_vulkan_headers",
        integrity = "sha256-17hHEvhGlle6o3pDbRoj778KY1T8iDW2dY7wNuFdzBQ=",
        strip_prefix = "Vulkan-Headers-1.4.349",
        urls = [
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.349.tar.gz",
        ],
    )

    http_archive(
        name = "ai_ml_sdk_vgf_library",
        build_file = "@//third_party/arm_vulkan_ml/ai-ml-sdk-vgf-library:BUILD.ai_ml_sdk_vgf_library",
        # This hard codes a flatbuffers version check that doesn't match
        # LiteRT's workspace dependency pin so a patch file is needed.
        patches = ["@//third_party/arm_vulkan_ml/ai-ml-sdk-vgf-library:PATCH.ai_ml_sdk_vgf_library"],
        sha256 = "9987b5a3f549dfb28b1b7da55ebcf77fc1f1686743e0de2084690b5c58155f80",
        strip_prefix = "ai-ml-sdk-vgf-library-0.11.0",
        urls = [
            "https://github.com/arm/ai-ml-sdk-vgf-library/archive/refs/tags/v0.11.0.tar.gz",
        ],
    )

    git_repository(
        name = "model_converter",
        commit = "a756a12845f7de80ad9164e53f7e2402659a8b96",  # v0.11.0
        patch_args = ["-p1"],
        patches = [
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/model_converter:0001-in-memory-c-api.patch",
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/model_converter:0002-c-api-only-dependencies.patch",
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/model_converter:0003-native-bazel-build.patch",
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/model_converter:0004-current-llvm-call-interface.patch",
        ],
        repo_mapping = {
            "@ai_ml_sdk_vgf_library": "@ai_ml_sdk_vgf_library",
            "@flatbuffers": "@flatbuffers",
            "@llvm-project": "@llvm-project",
            "@rules_cc": "@rules_cc",
        },
        remote = "https://github.com/arm/ai-ml-sdk-model-converter.git",
    )

    git_repository(
        name = "tosa-converter-for-tflite",
        commit = "5ebebc0baa524e8ff656bfad8be77f92b3ea8914",  # main, 2026-10-01
        patch_args = ["-p1"],
        patches = [
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/tcft:0001-size-optimized-c-shared-library.patch",
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/tcft:ats-singleop.patch",
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/tcft:operator-lowering.patch",
            "//litert/vendors/arm_vulkan_ml/compiler/external_deps/patches/tcft:float-operator-lowering.patch",
        ],
        remote = "https://gitlab.arm.com/tosa/tosa-converter-for-tflite.git",
    )
