# Copyright 2024 Google LLC.
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

load("//tflite/experimental/litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "litert_dispatch_device_context.h",
        "litert_dispatch_invocation_context.h",
    ],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + [
        "-Wl,-soname=libLiteRtDispatch_GoogleTensor.so",
        "-Wl,-lc++abi",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Mediatek.so",
    tags = [
        # Remove when sdk is available to bazel.
        "nobuilder",
        "notap",
    ],
    visibility = ["//tflite/experimental/litert:__subpackages__"],
    deps = [
        "//tflite/experimental/litert/c:litert_runtime_c_api_shared_lib",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/core:dynamic_loading",
        "//tflite/experimental/litert/core/util:tensor_type_util",
        "//tflite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "//tflite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "@org_tensorflow//third_party/neuro_pilot:latest_host_headers",
    ],
)

# This is cc_library target for `libLiteRtDispatch_Mediatek.so`.
cc_library(
    name = "dispatch_api_shared_lib",
    srcs = [":dispatch_api_so"],
)

cc_test(
    name = "dispatch_api_mediatek_test",
    srcs = [
        "dispatch_api_mediatek_test.cc",
    ],
    data = [
        ":dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "no-remote-exec",
        "no_oss",
        "nobuilder",
        "nosan",
        "notap",
    ],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_any",
        "//tflite/experimental/litert/core:filesystem",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:simple_model_npu",
        "//tflite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        # copybara:uncomment "@org_tensorflow//third_party/neuro_pilot:latest_host_headers",
    ],
)
