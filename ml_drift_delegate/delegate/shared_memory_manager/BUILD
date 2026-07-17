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
load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(
    default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"],
    default_visibility = ["//ml_drift_delegate:__subpackages__"],
)

cc_library(
    name = "shared_memory_manager",
    srcs = ["shared_memory_manager.cc"],
    hdrs = ["shared_memory_manager.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/genai/inference:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        "//ml_drift_delegate/delegate:unowned_tensor_desc",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor_utils_no_eigen",
        "//third_party/FP16",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/kernels:conv_apple_mpp",
        "@ml_drift//ml_drift/common/kernels:conv_generic",
        "@ml_drift//ml_drift/common/kernels:conv_wave_matrix",
        "@ml_drift//ml_drift/common/kernels:conv_wave_matrix_mali",
        "@ml_drift//ml_drift/common/kernels:conv_wave_memory",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/selectors:convolution_selector",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "shared_memory_manager_cl",
    hdrs = ["shared_memory_manager_cl.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:tensor",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "shared_memory_manager_cl_litert",
    srcs = ["shared_memory_manager_cl_litert.cc"],
    hdrs = ["shared_memory_manager_cl_litert.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_runtime_context_header",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "//third_party/odml/litert/weight_loader:external_weight_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:tensor",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@opencl_headers",
    ],
)

cc_test(
    name = "shared_memory_manager_cl_test",
    srcs = ["shared_memory_manager_cl_test.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":shared_memory_manager",
        ":shared_memory_manager_cl",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main_no_heapcheck",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "shared_memory_manager_webgpu_common",
    srcs = ["shared_memory_manager_webgpu_common.cc"],
    hdrs = ["shared_memory_manager_webgpu_common.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:executor",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
        "@ml_drift//ml_drift/webgpu:webgpu_headers",
    ],
)

cc_library(
    name = "shared_memory_manager_webgpu",
    srcs = ["shared_memory_manager_webgpu.cc"],
    hdrs = ["shared_memory_manager_webgpu.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        ":shared_memory_manager_webgpu_common",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "@com_google_absl//absl/status",
        "@ml_drift//ml_drift/common:executor",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
    ],
)

cc_library(
    name = "shared_memory_manager_webgpu_litert",
    srcs = ["shared_memory_manager_webgpu_litert.cc"],
    hdrs = ["shared_memory_manager_webgpu_litert.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        ":shared_memory_manager_webgpu_common",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_runtime_context_header",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "//third_party/odml/litert/weight_loader:external_weight_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:executor",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
    ],
)

cc_test(
    name = "shared_memory_manager_webgpu_test",
    srcs = ["shared_memory_manager_webgpu_test.cc"],
    tags = [
        "noasan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":shared_memory_manager",
        ":shared_memory_manager_webgpu",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/c:common",
        "//third_party/vulkan_loader",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main_no_heapcheck",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu/testing:webgpu_test",
    ],
)

cc_library(
    name = "shared_memory_manager_vulkan",
    hdrs = ["shared_memory_manager_vulkan.h"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        "//ml_drift_delegate/delegate:shared_vulkan_env",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/syrtis:environment",
        "@ml_drift//ml_drift/syrtis:vulkan_spatial_tensor",
    ],
)

cc_test(
    name = "shared_memory_manager_vulkan_test",
    srcs = ["shared_memory_manager_vulkan_test.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":shared_memory_manager",
        ":shared_memory_manager_vulkan",
        "//ml_drift_delegate/delegate:shared_vulkan_env",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/c:common",
        "//third_party/vulkan_loader",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main_no_heapcheck",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/syrtis/testing:vulkan_test",
    ],
)

objc_library(
    name = "shared_memory_manager_metal",
    hdrs = ["shared_memory_manager_metal.h"],
    aspect_hints = ["//tools/build_defs/swift:no_module"],
    copts = ["-ObjC++"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # Dependency placeholder: apple_frameworks Metal,
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/metal:metal_device",
        "@ml_drift//ml_drift/metal:metal_spatial_tensor",
        "//third_party/odml/infra/ml_drift_delegate:util",
        "//litert/c/internal:litert_runtime_context_header",
        "//ml_drift_delegate/delegate/serialization_weight_cache",
        "//third_party/odml/litert/weight_loader:external_weight_loader",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
    ],
)

objc_library(
    name = "shared_memory_manager_metal_test",
    testonly = 1,
    srcs = ["shared_memory_manager_metal_test.mm"],
    visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":shared_memory_manager",
        ":shared_memory_manager_metal",
        "@com_google_absl//absl/container:flat_hash_map",
        # Dependency placeholder: apple_frameworks Metal,
        # Dependency placeholder: apple_frameworks XCTest,
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/metal:metal_device",
        "//ml_drift_delegate/tflite:shared_const_tensor_map",
        "//tflite/c:common",
    ],
)
