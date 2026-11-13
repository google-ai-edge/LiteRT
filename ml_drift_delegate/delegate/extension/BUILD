# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GPU kernels added in this delegate on top of the kernels ML Drift provides.
# See README.md. Add one cc_library per kernel.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
    ],
)

cc_library(
    name = "scatter_nd_kernel",
    srcs = ["scatter_nd_kernel.cc"],
    hdrs = ["scatter_nd_kernel.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_test(
    name = "scatter_nd_gpu_test",
    srcs = ["scatter_nd_gpu_test.cc"],
    env = {
        "LITERT_DISABLE_OPENCL": "1",
    },
    tags = [
        "requires-gpu-sm90-full",  # Vulkan/WebGPU: no driver on MIG
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_gpu_options",
        "//tflite:version",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ] + litert_gpu_accelerator_deps(),
)
