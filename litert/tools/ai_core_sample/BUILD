# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//litert:__subpackages__"],
)

# Reference implementation of the caller (AICore) side of
# LiteRtCreateModelFromFdWithDmaContextCallback. Not part of LiteRT's
# supported API surface -- a real integrator (e.g. AICore) would replace
# this with its own allocator (a per-request VM-donated DMA-BUF, etc.).
cc_library(
    name = "dma_buf_provider",
    srcs = ["dma_buf_provider.cc"],
    hdrs = ["dma_buf_provider.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
    ],
)
