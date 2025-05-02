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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # Temporary visibility for GenAI until they bundle libLiteRtRuntimeCApi.so.
        "//third_party/odml/infra/genai/inference:__subpackages__",
        "//litert/c:__subpackages__",
        "//litert/compiler:__subpackages__",
        "//litert/core:__subpackages__",
        "//litert/runtime:__subpackages__",
        "//litert/tools:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
        "//third_party/odml/litert_lm/runtime:__subpackages__",
    ],
)

cc_library(
    name = "litert_environment",
    hdrs = ["//litert/cc:litert_environment.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/cc:litert_any",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_event",
    hdrs = ["//litert/cc:litert_event.h"],
    deps = [
        "//litert/c:litert_event",
        "//litert/c:litert_event_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = [
        "//litert/cc:litert_tensor_buffer.cc",
    ],
    hdrs = [
        "//litert/cc:litert_tensor_buffer.h",
        "//litert/cc:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_event",
        "//litert/c:litert_common",
        "//litert/c:litert_event",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//tflite/c:c_api_types",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_library(
    name = "litert_tensor_buffer_requirements",
    hdrs = [
        "//litert/cc:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//tflite/c:c_api_types",
        "@com_google_absl//absl/types:span",
    ],
)

# Dispatch Delegate of LiteRt.
# Warning: This API is not ABI stable and is subject to change.
cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "//litert/cc:litert_dispatch_delegate.h",
    ],
    deps = [
        "//litert/c:litert_environment_options",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["//litert/cc:litert_compiled_model.cc"],
    hdrs = ["//litert/cc:litert_compiled_model.h"],
    deps = [
        ":litert_custom_op_kernel",
        ":litert_environment",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_environment",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_options",
    hdrs = ["//litert/cc:litert_options.h"],
    deps = [
        ":litert_custom_op_kernel",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "litert_custom_op_kernel",
    srcs = ["//litert/cc:litert_custom_op_kernel.cc"],
    hdrs = ["//litert/cc:litert_custom_op_kernel.h"],
    deps = [
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_layout",
        "//litert/c:litert_logging",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_layout",
        "@com_google_absl//absl/types:span",
    ],
)

exports_files(srcs = ["litert_dispatch_delegate.h"])
