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

cc_import(
    name = "tensorrt_rtx_shared",
    shared_library = "lib/libtensorrt_rtx.so",
)

cc_library(
    name = "tensorrt_rtx_headers",
    hdrs = glob(["include/Nv*.h"]),
    includes = ["include"],
    deps = ["@local_cuda//:cuda_headers"],
)

cc_library(
    name = "tensorrt_rtx_stub",
    deps = [
        ":tensorrt_rtx_headers",
        ":tensorrt_rtx_shared",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "tensorrt",
    deps = [
        ":tensorrt_rtx_headers",
        ":tensorrt_rtx_stub",
    ],
)
