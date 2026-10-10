# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "tensor_desc",
    hdrs = ["tensor_desc.h"],
    deps = ["@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers"],
)

cc_library(
    name = "graph_desc",
    hdrs = ["graph_desc.h"],
    deps = [":tensor_desc"],
)

cc_library(
    name = "vgf_loader",
    srcs = ["vgf/loader.cc"],
    hdrs = ["vgf/loader.h"],
    deps = [
        ":graph_desc",
        "//litert/cc:litert_expected",
        "@ai_ml_sdk_vgf_library//:vgf",
        "@ai_ml_sdk_vgf_library//:vgf_utils",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
    ],
)
