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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "bytecode",
    srcs = ["bytecode.cc"],
    hdrs = ["bytecode.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:fingerprint",
    ],
)

cc_library(
    name = "memory_profile",
    srcs = ["memory_profile.cc"],
    hdrs = ["memory_profile.h"],
    deps = [
        "//litert/c/internal:litert_logging",
        "@com_google_absl//absl/time",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_test(
    name = "bytecode_test",
    srcs = ["bytecode_test.cc"],
    deps = [
        ":bytecode",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "profiling_test",
    srcs = ["profiling_test.cc"],
    tags = ["manual"],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":memory_profile",
        "//litert/vendors/nvidia/dispatch:dispatch_profiler",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensorrt_logger",
    hdrs = ["tensorrt_logger.h"],
    deps = [
        "//litert/c/internal:litert_logging",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)
