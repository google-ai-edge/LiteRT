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

load("//litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps")
load("//litert/build_common:special_rule.bzl", "gles_deps", "gles_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "event",
    srcs = ["event.cc"],
    hdrs = ["event.h"],
    deps = [
        ":gpu_environment",
        "//litert/c:litert_common",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "@com_google_absl//absl/strings:str_format",
        "@opencl_headers",
    ],
)

cc_library(
    name = "metrics",
    hdrs = ["metrics.h"],
    deps = ["//litert/c:litert_any"],
)

cc_library(
    name = "tensor_buffer",
    srcs = [
        "ahwb_buffer.cc",
        "dmabuf_buffer.cc",
        "fastrpc_buffer.cc",
        "gl_buffer.cc",
        "gl_texture.cc",
        "ion_buffer.cc",
        "open_cl_memory.cc",
        "open_cl_sync.cc",
        "tensor_buffer.cc",
        "tensor_buffer_requirements.cc",
    ],
    hdrs = [
        "ahwb_buffer.h",
        "dmabuf_buffer.h",
        "event.h",
        "fastrpc_buffer.h",
        "gl_buffer.h",
        "gl_texture.h",
        "ion_buffer.h",
        "open_cl_memory.h",
        "open_cl_sync.h",
        "tensor_buffer.h",
        "tensor_buffer_requirements.h",
        "//litert/c:litert_tensor_buffer.h",
    ],
    linkopts = gles_linkopts(),
    deps = [
        ":gpu_environment",
        "//litert/c:litert_common",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_layout",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer_utils",
        "//litert/cc/internal:litert_event",
        "//litert/core:environment",
        "//litert/core/util:tensor_type_util",
        "//litert/core/util:to_string",
        "//tflite/delegates/gpu/cl:buffer",
        "//tflite/delegates/gpu/cl:cl_command_queue",
        "//tflite/delegates/gpu/cl:cl_context",
        "//tflite/delegates/gpu/cl:cl_memory",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "//tflite/delegates/gpu/cl:tensor",
        "//tflite/delegates/gpu/cl:util",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_deps() + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:egl_environment",
            "//tflite/delegates/gpu/gl:gl_buffer",
            "//tflite/delegates/gpu/gl:gl_texture",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "gpu_environment_header",
    hdrs = ["gpu_environment.h"],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//tflite/delegates/gpu/cl:cl_command_queue",
        "//tflite/delegates/gpu/cl:cl_context",
        "//tflite/delegates/gpu/cl:cl_device",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "@opencl_headers",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/cl:gl_interop",
            "//tflite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "gpu_environment",
    srcs = ["gpu_environment.cc"],
    hdrs = ["gpu_environment.h"],
    linkopts = gles_linkopts(),
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//tflite/delegates/gpu/cl:cl_command_queue",
        "//tflite/delegates/gpu/cl:cl_context",
        "//tflite/delegates/gpu/cl:cl_device",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "@opencl_headers",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/cl:gl_interop",
            "//tflite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "gpu_environment_test",
#     srcs = ["gpu_environment_test.cc"],
#     linkopts = select({
#         "@org_tensorflow//tensorflow:android": ["-landroid"],
#         "//conditions:default": [],
#     }),
#     tags = ["requires-gpu-nvidia"],
#     deps = [
#         ":gpu_environment",
#         "@com_google_googletest//:gtest_main",
#         "@ml_drift//ml_drift/cl:environment",
#         "@ml_drift//ml_drift/cl:opencl_wrapper",
#         "//litert/c:litert_any",
#         "//litert/c:litert_environment_options",
#         "//litert/cc:litert_any",
#         "//litert/core:environment",
#         "//litert/test:matchers",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "tfl_utils",
    srcs = ["tfl_utils.cc"],
    hdrs = ["tfl_utils.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "external_litert_buffer_context",
    srcs = ["external_litert_buffer_context.cc"],
    hdrs = ["external_litert_buffer_context.h"],
    deps = [
        ":tfl_utils",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/cc/internal:litert_tensor_buffer_requirements",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# external_litert_buffer_context target depending on litert_runtime_c_api_shared_lib.
cc_library(
    name = "external_litert_buffer_context_runtimecapi",
    srcs = ["external_litert_buffer_context.cc"],
    hdrs = ["external_litert_buffer_context.h"],
    deps = [
        ":tfl_utils",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_requirements",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "compiled_model",
    srcs = ["compiled_model.cc"],
    hdrs = ["compiled_model.h"],
    deps = [
        ":accelerator",
        ":custom_op_dispatcher",
        ":external_litert_buffer_context",
        ":litert_cpu_options",
        ":litert_runtime_options",
        ":metrics",
        ":profiler",
        ":tensor_buffer",
        "//litert/c:litert_accelerator",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_profiler_event",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_tensor_buffer_utils",
        "//litert/cc/internal:litert_event",
        "//litert/cc/internal:litert_options",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/cc/internal:litert_tensor_buffer_requirements",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core:build_stamp",
        "//litert/core:environment",
        "//litert/core:options",
        "//litert/core/model",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime/dispatch:dispatch_opaque_options",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:model_builder",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types_opaque_delegate",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/core/api",
        "//tflite/delegates/utils:simple_opaque_delegate",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:allocation",
    ],
)

cc_test(
    name = "compiled_model_test",
    srcs = ["compiled_model_test.cc"],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    # require GPU to run OpenCL tests.
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":compiled_model",
        ":tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_options",
        "//litert/c:litert_profiler",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gl_buffer_test",
    srcs = ["gl_buffer_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-landroid",
        ],
        "//conditions:default": [],
    }),
    tags = ["notap"],
    deps = [
        ":gpu_environment_header",
        ":tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_event",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_platform_support",
        "//litert/cc:litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/test:matchers",
        "@com_google_absl//absl/types:span",
    ] + gtest_main_no_heapcheck_deps() + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:egl_environment",
            "//tflite/delegates/gpu/gl:gl_buffer",
        ],
        "//conditions:default": [],
    }),
)

litert_device_exec(
    name = "gl_buffer_device_test",
    data = [],
    target = ":gl_buffer_test",
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "tensor_buffer_conversion",
#     srcs = ["tensor_buffer_conversion.cc"],
#     hdrs = ["tensor_buffer_conversion.h"],
#     linkopts = gles_linkopts(),
#     deps = [
#         ":tensor_buffer",
#         "@com_google_absl//absl/strings:str_format",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc:litert_expected",
#         "//litert/cc:litert_macros",
#         "//litert/cc:litert_tensor_buffer_utils",
#     ] + gles_deps(),
# )
#
# cc_test(
#     name = "tensor_buffer_conversion_test",
#     srcs = ["tensor_buffer_conversion_test.cc"],
#     linkopts = select({
#         "@org_tensorflow//tensorflow:android": ["-landroid"],
#         "//conditions:default": [],
#     }),
#     tags = ["notap"],
#     deps = [
#         ":tensor_buffer",
#         ":tensor_buffer_conversion",
#         "@com_google_googletest//:gtest_main",
#         "//litert/c:litert_common",
#         "//litert/c:litert_model",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc:litert_expected",
#         "//litert/cc:litert_layout",
#         "//litert/cc:litert_macros",
#         "//litert/core:environment",
#         "//litert/test:matchers",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "accelerator",
    hdrs = ["accelerator.h"],
    deps = [
        ":metrics",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
    ],
)

cc_library(
    name = "accelerator_registry",
    srcs = ["accelerator_registry.cc"],
    hdrs = ["accelerator_registry.h"],
    deps = [
        ":accelerator",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_shared_library",
    ],
)

cc_library(
    name = "custom_op_dispatcher",
    srcs = ["custom_op_dispatcher.cc"],
    hdrs = ["custom_op_dispatcher.h"],
    deps = [
        ":external_litert_buffer_context",
        ":tfl_utils",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_layout",
        "//litert/c:litert_logging",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/core:options",
        "//tflite:builtin_ops",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_runtime_options",
    hdrs = ["litert_runtime_options.h"],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        "//litert/c:litert_profiler_event",
        "//tflite/core/api",
        "//tflite/profiling:profile_buffer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "litert_cpu_options",
    hdrs = ["litert_cpu_options.h"],
    deps = ["//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only"],
)

cc_library(
    name = "litert_google_tensor",
    hdrs = ["litert_google_tensor.h"],
    deps = ["//litert/c/options:litert_google_tensor_options_type"],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    deps = [
        ":profiler",
        "//litert/c:litert_logging",
        "//litert/c:litert_profiler_event",
        "//tflite/core/api",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
