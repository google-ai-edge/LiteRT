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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//third_party/nvidia_cuda:build_defs.bzl", "cuda_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

# The decode step of a Gemma 4 mixture-of-experts model in one persistent
# kernel, from a program of instructions (program.h).
cuda_library(
    name = "megakernel_cuda",
    srcs = ["megakernel.cu"],
    hdrs = [
        "megakernel.h",
        "program.h",
        "block_sync.cuh",
        "decode_attention.cuh",
        "int4_block_gemv.cuh",
        "qkv_norm_rope.cuh",
        # The per-op launches' C interfaces: the arena, the workspace sizes.
        "//litert/vendors/nvidia/trtllm:decode_attention.h",
        "//litert/vendors/nvidia/trtllm:int4_block_gemv.h",
    ],
    # copybara:uncomment_begin(google-only)
    # # The architectures of trtllm:gemma4_head_kernels_cuda.
    # copts = [
    # "--no-cuda-gpu-arch=sm_60",
    # "--no-cuda-gpu-arch=sm_70",
    # "--cuda-gpu-arch=sm_120",
    # ],
    # copybara:uncomment_end
    # copybara:uncomment linkstatic = 1,
    tags = ["manual"],
    deps = ["@local_cuda//:cuda_runtime"],
)

cc_import(
    name = "megakernel_archive",
    static_library = ":megakernel_cuda",
    tags = ["manual"],
    alwayslink = 1,
)

# The megakernel and the reference step (today's per-op launches).
cc_library(
    name = "megakernel",
    srcs = ["megakernel_reference.cc"],
    hdrs = [
        "megakernel.h",
        "program.h",
    ],
    tags = ["manual"],
    deps = [
        ":megakernel_archive",
        ":megakernel_steps",
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

# A launch's trace as Chrome trace JSON and a summary (mktrace).
cc_library(
    name = "mktrace",
    srcs = ["mktrace.cc"],
    hdrs = ["mktrace.h"],
    tags = ["manual"],
    deps = [":megakernel"],
)

cuda_library(
    name = "megakernel_lab_data_cuda",
    srcs = ["megakernel_lab_data.cu"],
    hdrs = ["megakernel_lab_data.h"],
    # copybara:uncomment linkstatic = 1,
    tags = ["manual"],
    deps = ["@local_cuda//:cuda_runtime"],
)

cc_import(
    name = "megakernel_lab_data_archive",
    static_library = ":megakernel_lab_data_cuda",
    tags = ["manual"],
)

# The lab: the reference and the megakernel on the same synthetic step,
# compared bit for bit and timed interleaved (megakernel_lab.cc).
cc_binary(
    name = "megakernel_lab",
    srcs = [
        "megakernel_lab.cc",
        "megakernel_lab_data.h",
    ],
    tags = ["manual"],
    deps = [
        ":megakernel",
        ":megakernel_lab_data_archive",
        ":mktrace",
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

# The megakernel against the reference step on a small synthetic step.
cc_test(
    name = "megakernel_test",
    srcs = [
        "megakernel_lab_data.h",
        "megakernel_test.cc",
    ],
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
        ":megakernel",
        ":megakernel_lab_data_archive",
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "megakernel_steps_cuda",
    srcs = ["megakernel_steps.cu"],
    hdrs = ["megakernel_steps.h"],
    # copybara:uncomment linkstatic = 1,
    tags = ["manual"],
    deps = ["@local_cuda//:cuda_runtime"],
)

cc_import(
    name = "megakernel_steps_archive",
    static_library = ":megakernel_steps_cuda",
    tags = ["manual"],
)

cc_library(
    name = "megakernel_steps",
    hdrs = ["megakernel_steps.h"],
    tags = ["manual"],
    deps = [
        ":megakernel_steps_archive",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

# The decode step of a Gemma 4 mixture-of-experts model as one plugin layer
# that runs it as a program.
cc_library(
    name = "megakernel_plugin",
    srcs = ["megakernel_plugin.cc"],
    hdrs = ["megakernel_plugin.h"],
    tags = ["manual"],
    deps = [
        ":megakernel",
        ":megakernel_steps",
        "//litert/vendors/nvidia/compiler:moe_plugin",
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@com_google_absl//absl/types:span",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)

# The compiler's side: which subgraph is taken whole, and its lowering to the
# plugin layer, which compiler/tensorrt_graph_builder.cc includes inside its
# builder class (megakernel_lowering.inc).
cc_library(
    name = "megakernel_lowering",
    srcs = ["megakernel_lowering.cc"],
    hdrs = ["megakernel_lowering.h"],
    tags = ["manual"],
    textual_hdrs = ["megakernel_lowering.inc"],
    deps = [
        ":megakernel_plugin",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/compiler/cc:litert_model",
        "@com_google_absl//absl/types:span",
    ],
)
