# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//third_party/nvidia_cuda:build_defs.bzl", "cuda_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cuda_library(
    name = "gemma4_head_kernels_cuda",
    srcs = [
        "decode_attention.cu",
        "gated_delta.cu",
        "gated_delta_net.cu",
        "global_attention.cu",
        "head_kernels.cu",
        "int2_gemv.cu",
        "int4_block_gemv.cu",
        "qkv_norm_rope.cu",
        "quant_cache.cu",
        "quant_gemm.cu",
        "quant_linear.cu",
        "subbyte_gemm.cu",
        "subbyte_gemm_int8.cu",
        "tiled_attention.cu",
    ],
    hdrs = [
        "decode_attention.h",
        "gated_delta.h",
        "gated_delta_net.h",
        "global_attention.h",
        "head_kernels.h",
        "int2_gemv.h",
        "int4_block_gemv.cuh",
        "int4_block_gemv.h",
        "qkv_norm_rope.h",
        "quant_cache.h",
        "quant_gemm.h",
        "quant_linear.h",
        "subbyte_gemm.h",
        "tiled_attention.h",
    ],
    # Tensor-core instructions require SM80; also include RTX Blackwell kernels.
    # copybara:uncomment_begin(google-only)
    # copts = [
    # "--no-cuda-gpu-arch=sm_60",
    # "--no-cuda-gpu-arch=sm_70",
    # "--cuda-gpu-arch=sm_120",
    # ],
    # copybara:uncomment_end
    linkstatic = 1,
    tags = ["manual"],
    deps = [
        "@local_cuda//:cuda_runtime",
    ],
)

cc_import(
    name = "gemma4_head_kernels_archive",
    static_library = ":gemma4_head_kernels_cuda",
    tags = ["manual"],
    alwayslink = 1,
)

cc_library(
    name = "gemma4_head_kernels",
    hdrs = [
        "decode_attention.h",
        "gated_delta.h",
        "gated_delta_net.h",
        "global_attention.h",
        "head_kernels.h",
        "int2_gemv.h",
        "int4_block_gemv.h",
        "qkv_norm_rope.h",
        "quant_cache.h",
        "quant_gemm.h",
        "quant_linear.h",
        "subbyte_gemm.h",
        "tiled_attention.h",
    ],
    tags = ["manual"],
    deps = [
        ":gemma4_head_kernels_archive",
        "@local_cuda//:cuda_headers",
    ],
)

# Host references of the quant_linear kernels, for tests.
cc_library(
    name = "quant_linear_reference",
    testonly = 1,
    hdrs = ["quant_linear_reference.h"],
    tags = ["manual"],
    deps = [":gemma4_head_kernels"],
)

cc_test(
    name = "quant_linear_test",
    srcs = ["quant_linear_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        ":quant_linear_reference",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "quant_gemm_test",
    srcs = ["quant_gemm_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        ":quant_linear_reference",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "qkv_norm_rope_test",
    srcs = ["qkv_norm_rope_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "int4_block_gemv_test",
    srcs = ["int4_block_gemv_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "gated_delta_test",
    srcs = ["gated_delta_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:linux_aarch64": [],
        "//litert:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "gated_delta_net_test",
    srcs = ["gated_delta_net_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:linux_aarch64": [],
        "//litert:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "int2_gemv_test",
    srcs = ["int2_gemv_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "gemma4_head_kernels_test",
    srcs = ["gemma4_head_kernels_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "//litert/vendors/nvidia:bytecode",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "decode_attention_benchmark_test",
    srcs = ["decode_attention_benchmark_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "global_attention_test",
    srcs = ["global_attention_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "quant_attention_reference",
    testonly = True,
    hdrs = ["quant_attention_reference.h"],
    tags = ["manual"],
)

cc_test(
    name = "quant_attention_test",
    srcs = ["quant_attention_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        ":quant_attention_reference",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "tiled_attention_test",
    srcs = ["tiled_attention_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "quant_linear_benchmark_test",
    srcs = ["quant_linear_benchmark_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "subbyte_gemm_benchmark_test",
    srcs = ["subbyte_gemm_benchmark_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "subbyte_gemm_int8_test",
    srcs = ["subbyte_gemm_int8_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "subbyte_gemm_test",
    srcs = ["subbyte_gemm_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

# Qwen 4's sparse attention (QSA): the indexer and the sparse attention of
# odml.qsa_indexer_update and odml.sparse_sdpa_transposed. Their own archive,
# so the kernels above do not rebuild with them.
cuda_library(
    name = "qsa_kernels_cuda",
    srcs = [
        "qsa_indexer.cu",
        "qsa_sparse_attention.cu",
    ],
    hdrs = [
        "qsa_common.cuh",
        "qsa_indexer.h",
        "qsa_sparse_attention.h",
    ],
    # copybara:uncomment_begin(google-only)
    # copts = [
    # "--no-cuda-gpu-arch=sm_60",
    # "--no-cuda-gpu-arch=sm_70",
    # "--cuda-gpu-arch=sm_120",
    # ],
    # copybara:uncomment_end
    linkstatic = 1,
    tags = ["manual"],
    deps = [
        "@local_cuda//:cuda_runtime",
    ],
)

cc_import(
    name = "qsa_kernels_archive",
    static_library = ":qsa_kernels_cuda",
    tags = ["manual"],
    alwayslink = 1,
)

cc_library(
    name = "qsa_kernels",
    hdrs = [
        "qsa_indexer.h",
        "qsa_sparse_attention.h",
    ],
    tags = ["manual"],
    deps = [
        ":qsa_kernels_archive",
        "@local_cuda//:cuda_headers",
    ],
)

# Host references of the QSA kernels, for tests.
cc_library(
    name = "qsa_reference",
    testonly = 1,
    hdrs = ["qsa_reference.h"],
    tags = ["manual"],
)

cc_test(
    name = "qsa_kernels_test",
    srcs = ["qsa_kernels_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:windows": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":qsa_kernels",
        ":qsa_reference",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)
