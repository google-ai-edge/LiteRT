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

package(default_visibility = ["//visibility:public"])

exports_files(["bin/nvcc"])

filegroup(
    name = "headers",
    srcs = glob(["include/**"]),
)

# Inputs that nvcc launches internally. Declaring these keeps the standalone
# CUDA archive rule compatible with Bazel's sandbox.
filegroup(
    name = "nvcc_tools",
    srcs = [
        "bin/cudafe++",
        "bin/fatbinary",
        "bin/nvcc",
        "bin/ptxas",
        "nvvm/bin/cicc",
        "nvvm/libdevice/libdevice.10.bc",
    ] + glob(["bin/crt/**"]),
)

cc_library(
    name = "cuda_headers",
    hdrs = [":headers"],
    includes = ["include"],
)

cc_import(
    name = "cudart_shared",
    shared_library = "lib64/libcudart.so",
)

cc_library(
    name = "cuda_runtime",
    deps = [":cudart_shared"],
)

cc_library(
    name = "cuda",
    deps = [
        ":cuda_headers",
        ":cuda_runtime",
    ],
)
