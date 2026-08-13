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
        "head_kernels.cu",
        "int2_gemv.cu",
    ],
    hdrs = [
        "head_kernels.h",
        "int2_gemv.h",
    ],
    linkstatic = 1,
    deps = [
        "@local_cuda//:cuda_runtime",
    ],
)

cc_import(
    name = "gemma4_head_kernels_archive",
    static_library = ":gemma4_head_kernels_cuda",
    alwayslink = 1,
)

cc_library(
    name = "gemma4_head_kernels",
    hdrs = [
        "head_kernels.h",
        "int2_gemv.h",
    ],
    deps = [
        ":gemma4_head_kernels_archive",
        "@local_cuda//:cuda_headers",
    ],
)

cc_test(
    name = "gemma4_head_kernels_test",
    srcs = ["gemma4_head_kernels_test.cc"],
    tags = [
        "exclusive",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
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
