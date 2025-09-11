# Copyright 2025 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # Temporary visibility for GenAI until they bundle libLiteRtRuntimeCApi.so.
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # "//litert/c:__pkg__",
        # "//litert/cc:__pkg__",
        # "//litert/integration_test:__pkg__",
        # "//litert/test:__pkg__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = ["litert_dispatch_delegate.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
    ],
)

# Same as :litert_tensor_buffer, but without the registry dependency.
# This is for accelerator shared libs not to depend on the registry, but depend on CAPI.so instead.
cc_library(
    name = "litert_tensor_buffer_without_registry",
    srcs = ["//litert/cc:litert_tensor_buffer.cc"],
    hdrs = [
        "//litert/cc:litert_tensor_buffer.h",
        "//litert/cc:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_event",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_library(
    name = "litert_external_litert_buffer_context",
    hdrs = ["litert_external_litert_buffer_context.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_external_litert_buffer_context",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:common",
    ],
)

exports_files(["litert_dispatch_delegate.h"])
