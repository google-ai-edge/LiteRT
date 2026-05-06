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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps")
load(
    "//litert/build_common:special_rule.bzl",
    "gles_deps",
    "gles_linkopts",
    "litert_android_linkopts",
    "litert_metal_deps",
    "litert_metal_deps_without_gpu_environment",
    "litert_metal_linkopts",
    "litert_metal_opts",
)
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    # copybara:uncomment_begin(oss litert)
    # default_visibility = ["//litert:litert_internal_users"],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

config_setting(
    name = "litert_enable_fabric_integration",
    values = {"copt": "-DLITERT_ENABLE_FABRIC_INTEGRATION"},
)

cc_library(
    name = "event",
    srcs = ["event.cc"],
    hdrs = ["event.h"],
    copts = litert_metal_opts(),
    linkopts = litert_metal_linkopts(),
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_opencl_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "@com_google_absl//absl/strings:str_format",
    ] + select({
        "//litert/build_common:build_include_gpu_enabled": [
            ":gpu_environment",
            "//tflite/delegates/gpu/cl:opencl_wrapper",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "metrics",
    hdrs = ["metrics.h"],
    deps = ["//litert/c:litert_any"],
)

cc_library(
    name = "gl_buffer",
    srcs = ["gl_buffer.cc"],
    hdrs = ["gl_buffer.h"],
    linkopts = gles_linkopts(),
    deps = [
        ":ahwb_buffer",
        ":gpu_environment_header",
        "//litert/c:litert_common",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ] + gles_deps() + select({
        "//litert/build_common:android_gpu_enabled": [
            "//tflite/delegates/gpu/gl:egl_environment",
            "//tflite/delegates/gpu/gl:gl_buffer",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "ahwb_wrapper",
    srcs = ["ahwb_wrapper.cc"],
    hdrs = ["ahwb_wrapper.h"],
)

cc_library(
    name = "ahwb_buffer",
    srcs = ["ahwb_buffer.cc"],
    hdrs = ["ahwb_buffer.h"],
    deps = [
        ":ahwb_wrapper",
        ":event",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_library(
    name = "dmabuf_buffer",
    srcs = ["dmabuf_buffer.cc"],
    hdrs = ["dmabuf_buffer.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "fastrpc_buffer",
    srcs = ["fastrpc_buffer.cc"],
    hdrs = ["fastrpc_buffer.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gl_texture",
    srcs = ["gl_texture.cc"],
    hdrs = ["gl_texture.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//tflite/delegates/gpu/gl:gl_texture",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "ion_buffer",
    srcs = ["ion_buffer.cc"],
    hdrs = ["ion_buffer.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "open_cl_memory",
    srcs = ["open_cl_memory.cc"],
    hdrs = ["open_cl_memory.h"],
    deps = [
        ":ahwb_buffer",
        ":gl_buffer",
        ":gpu_environment_header",
        ":open_cl_sync",
        ":tensor_buffer_lockstate",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/util:tensor_type_util",
        "//tflite/delegates/gpu/cl:buffer",
        "//tflite/delegates/gpu/cl:cl_context",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "//tflite/delegates/gpu/cl:util",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@opencl_headers",
    ],
)

cc_library(
    name = "open_cl_sync",
    srcs = ["open_cl_sync.cc"],
    hdrs = ["open_cl_sync.h"],
    deps = [
        ":gpu_environment_header",
        ":litert_gpu_util",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_macros",
        "//tflite/delegates/gpu/cl:cl_command_queue",
        "//tflite/delegates/gpu/cl:cl_memory",
        "//tflite/delegates/gpu/cl:tensor",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@opencl_headers",
    ],
)

RUNTIME_TENSOR_BUFFER_HEADER_DEPS = [
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/container:node_hash_map",
    "@com_google_absl//absl/types:span",
    "//litert/c:litert_common",
    "//litert/c:litert_event_type",
    "//litert/c:litert_gl_types",
    "//litert/c:litert_layout",
    "//litert/c:litert_model_types",
    "//litert/c:litert_tensor_buffer_types",
    "//litert/c:litert_webgpu_types",
    "//litert/cc:litert_expected",
    "//litert/cc:litert_macros",
    "@opencl_headers",
] + select({
    "//litert/build_common:android_gpu_enabled": [
        ":gl_buffer",
        ":gl_texture",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:gl_texture",
    ],
    "//conditions:default": [],
}) + select({
    "//litert/build_common:build_include_gpu_enabled": [
        ":open_cl_memory",
        ":open_cl_sync",
    ],
    "//conditions:default": [],
}) + select({
    "//litert/build_common:litert_disable_cpu": [],
    "//conditions:default": ["@XNNPACK"],
}) + select({
    "//litert/build_common:build_include_npu_enabled": [
        ":dmabuf_buffer",
        ":fastrpc_buffer",
        ":ion_buffer",
    ],
    "//conditions:default": [],
}) + select({
    "//litert:android": [
        ":ahwb_buffer",
    ],
    "//conditions:default": [],
}) + litert_metal_deps()

cc_library(
    name = "tensor_buffer_header",
    hdrs = [
        "custom_buffer.h",
        "event.h",
        "tensor_buffer.h",
        "tensor_buffer_lockstate.h",
        "tensor_buffer_requirements.h",
        "//litert/c:litert_custom_tensor_buffer.h",
        "//litert/c:litert_opencl_types.h",
        "//litert/c:litert_tensor_buffer.h",
        "//litert/c:litert_tensor_buffer_requirements.h",
    ],
    deps = RUNTIME_TENSOR_BUFFER_HEADER_DEPS,
)

cc_library(
    name = "tensor_buffer",
    srcs = [
        "custom_buffer.cc",
        "tensor_buffer.cc",
    ],
    hdrs = [
        "custom_buffer.h",
        "event.h",
        "tensor_buffer.h",
        "tensor_buffer_lockstate.h",
        "//litert/c:litert_custom_tensor_buffer.h",
        "//litert/c:litert_opencl_types.h",
        "//litert/c:litert_tensor_buffer.h",
    ],
    copts = litert_metal_opts() + select({
        "//litert/build_common:litert_disable_cpu": ["-DLITERT_DISABLE_CPU"],
        "//conditions:default": [],
    }),
    linkopts = litert_metal_linkopts(),
    deps = [
        ":event",
        ":tensor_buffer_registry",
        "//litert/c:litert_any",
        "//litert/c:litert_environment_header",
        "//litert/c:litert_environment_options_header",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c/internal:litert_tensor_buffer_registry_header",
        "//litert/cc/internal:litert_tensor_buffer_utils",
        "//litert/core:environment",
        "//litert/core/util:tensor_type_util",
        "//litert/core/util:to_string",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//litert/build_common:build_include_gpu_enabled": [
            ":gpu_environment",
        ],
        "//conditions:default": [],
    }) + RUNTIME_TENSOR_BUFFER_HEADER_DEPS,
)

cc_library(
    name = "tensor_buffer_requirements",
    srcs = [
        "tensor_buffer_requirements.cc",
    ],
    hdrs = [
        "tensor_buffer_requirements.h",
        "//litert/c:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/core/util:to_string",
    ],
)

cc_library(
    name = "tensor_buffer_lockstate",
    hdrs = ["tensor_buffer_lockstate.h"],
    deps = ["//litert/c:litert_common"],
)

cc_library(
    name = "tensor_buffer_registry",
    srcs = ["tensor_buffer_registry.cc"],
    hdrs = [
        "tensor_buffer_registry.h",
        "//litert/c:litert_custom_tensor_buffer.h",
    ],
    visibility = [
        "//litert/c:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/internal:litert_tensor_buffer_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tensor_buffer_registry_header",
    srcs = [],
    hdrs = [
        "tensor_buffer_registry.h",
        "//litert/c:litert_custom_tensor_buffer.h",
    ],
    visibility = [
        "//litert/core:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
    ],
)

cc_test(
    name = "tensor_buffer_registry_test",
    srcs = ["tensor_buffer_registry_test.cc"],
    deps = [
        ":tensor_buffer",
        ":tensor_buffer_registry",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_tensor_buffer_registry",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

GPU_ENVIRONMENT_DEPS = [
    "//litert/c:litert_any",
    "//litert/c:litert_common",
    "//litert/c:litert_gl_types",
    "//litert/c/internal:litert_logging",
    "//litert/cc:litert_expected",
    "//litert/cc:litert_macros",
    "//litert/core:environment_options",
    "@opencl_headers",
    "//tflite/delegates/gpu/cl:cl_command_queue",
    "//tflite/delegates/gpu/cl:cl_context",
    "//tflite/delegates/gpu/cl:cl_device",
    "//tflite/delegates/gpu/cl:opencl_wrapper",
] + select({
    "//litert:android": [
        "//tflite/delegates/gpu/cl:gl_interop",
        "//tflite/delegates/gpu/gl:egl_environment",
    ],
    "//conditions:default": [],
}) + gles_deps()

cc_library(
    name = "gpu_environment_header",
    hdrs = ["gpu_environment.h"],
    copts = litert_metal_opts(),
    linkopts = litert_metal_linkopts(),
    deps = GPU_ENVIRONMENT_DEPS + litert_metal_deps() + [
        "//litert/c:litert_environment_options_header",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_environment",
    srcs = ["gpu_environment.cc"],
    hdrs = ["gpu_environment.h"],
    copts = litert_metal_opts(),
    linkopts = gles_linkopts(),
    deps = GPU_ENVIRONMENT_DEPS + [
        "//litert/c:litert_environment_options_header",
        "//litert/cc:litert_any",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ] + litert_metal_deps(),
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "gpu_environment_test",
#     srcs = ["gpu_environment_test.cc"],
#     linkopts = litert_android_linkopts(),
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
        ":tensor_identifier",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/core:options",
        "//tflite:framework_stable",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tensor_identifier",
    hdrs = ["tensor_identifier.h"],
)

cc_library(
    name = "external_litert_buffer_context",
    srcs = ["external_litert_buffer_context.cc"],
    hdrs = ["external_litert_buffer_context.h"],
    copts = litert_metal_opts(),
    linkopts = litert_metal_linkopts(),
    deps = [
        ":tensor_buffer_header",
        ":tensor_buffer_requirements",
        ":tensor_identifier",
        ":tfl_utils",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "magic_number_utils",
    srcs = ["magic_number_utils.cc"],
    hdrs = ["magic_number_utils.h"],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools",
        "//tflite:framework",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "magic_number_utils_test",
    srcs = ["magic_number_utils_test.cc"],
    data = ["//litert/test:testdata/model_magic_test.tflite"],
    deps = [
        ":gpu_environment",
        ":magic_number_utils",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/core:environment",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "dispatch_runner_sb",
#     srcs = ["fabric/dispatch_runner_sb.cc"],
#     hdrs = ["fabric/dispatch_runner_sb.h"],
#     copts = ["-DLITERT_ENABLE_FABRIC_INTEGRATION"],
#     deps = [
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/container:flat_hash_set",
#         "@com_google_absl//absl/hash",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings:str_format",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//third_party/odml/infra/fabric/runtime:executor",
#         "//third_party/odml/infra/fabric/runtime:inference_program",
#         "//third_party/odml/infra/northbound/aot:buffer_management",
#         "//third_party/odml/infra/northbound/aot:emission_graph",
#         "//third_party/odml/infra/northbound/aot:hardware_buffer",
#         "//third_party/odml/infra/northbound/aot:program_analysis",
#         "//third_party/odml/infra/northbound/graph:identifiers",
#         "//third_party/odml/infra/southbound:sb_api_cpp",
#         "//litert/c:litert_layout",
#         "//litert/c:litert_options",
#         "//litert/c:litert_tensor_buffer",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/c/internal:litert_runtime_context_header",
#         "//litert/cc:litert_buffer_ref",
#         "//litert/core:environment",
#         "//litert/core:filesystem",
#         "//litert/vendors/c:litert_dispatch_c_api",
#     ],
# )
# copybara:uncomment_end

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "fabric_execution_contract_resolver",
#     srcs = ["fabric/execution_contract_resolver.cc"],
#     hdrs = ["fabric/execution_contract_resolver.h"],
#     deps = [
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings:str_format",
#         "@com_google_absl//absl/strings:string_view",
#         "//third_party/gloop/util/status:status_macros",
#         "//third_party/odml/infra/fabric/runtime:transmission_graph_cc_proto",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "compiled_model",
    srcs = [
        "compiled_model.cc",
        # copybara:uncomment "fabric/compiled_model_fabric_internal.cc",
    ],
    hdrs = [
        "compiled_model.h",
        # copybara:uncomment "fabric/compiled_model_fabric_internal.h",
        "stub_op_resolver.h",
    ],
    copts = litert_metal_opts() + select({
        "//litert:without_builtin_ops": [
            # No builtin ops. Xnnpack is used for CPU backend.
            "-DLITERT_NO_BUILTIN_OPS",
        ],
        "//litert/build_common:litert_disable_cpu": [
            # No support for CPU backend. Builtin ops are still needed for GPU.
            "-DLITERT_DISABLE_CPU",
        ],
        "//conditions:default": [],
    }),
    linkopts = litert_metal_linkopts(),
    deps = [
        ":accelerator",
        ":custom_op_dispatcher",
        ":external_litert_buffer_context",
        ":litert_runtime_options",
        ":magic_number_utils",
        ":metrics",
        ":profiler",
        ":tensor_buffer",
        ":tensor_buffer_requirements",
        ":tensor_identifier",
        ":tfl_utils",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c/options:litert_cpu_options",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_layout",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_profiler_event",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c/internal:litert_delegate_wrapper",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/cc/internal:scoped_weight_source",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc/internal:litert_consts",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_tensor_buffer_utils",
        "//litert/core:buffer_error_reporter",
        "//litert/core:build_stamp",
        "//litert/core:environment",
        "//litert/core:error_reporter",
        "//litert/core:options",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime/dispatch:dispatch_opaque_options",
        # copybara:uncomment "//third_party/odml/litert/weight_loader:external_weight_loader",
        "//weight_loader:external_weight_loader",  # copybara:comment
        "//tflite/converter:allocation",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:model_builder",
    ] +
    # copybara:uncomment_begin(google-only)
    # select({
    # ":litert_enable_fabric_integration": [
    # ":dispatch_runner_sb",
    # ":litert_fabric_options",
    # "//third_party/odml/infra/fabric/runtime:buffer_utils",
    # "//third_party/odml/infra/fabric/runtime:executor",
    # "//third_party/odml/infra/fabric/runtime:inference_program",
    # "//third_party/odml/infra/fabric/runtime:tg_proto_converter",
    # "//third_party/odml/infra/fabric/runtime:transmission_graph_cc_proto",
    # "//third_party/odml/infra/northbound/aot:hardware_buffer",
    # "//third_party/odml/infra/northbound/aot:program_analysis",
    # "//third_party/odml/infra/northbound/graph:identifiers",
    # "//litert/core:filesystem",
    # "//litert/vendors/c:litert_dispatch_c_api",
    # ],
    # "//conditions:default": [],
    # }) +
    # copybara:uncomment_end
    [
        "//tflite:mutable_op_resolver",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types_opaque_delegate",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/core/api",
        "//tflite/delegates/utils:simple_opaque_delegate",
        "//tflite/schema:schema_fbs",
    ] + select({
        "//litert/build_common:build_include_npu_enabled": [
            "//litert/compiler/plugin:compiler_plugin",
            "//litert/core/cache:compilation_cache",
            "//litert/core/model:model_serialize",
        ],
        "//conditions:default": [],
    }) + select({
        "//litert:without_builtin_ops": [
            # No builtin ops. Xnnpack is used for CPU backend.
            ":litert_cpu_options",
            "//litert/runtime/accelerators/xnnpack:xnnpack_accelerator",  # buildcleaner: keep
        ],
        "//litert/build_common:litert_disable_cpu": [
            # No support for CPU backend. Builtin ops are still needed for GPU.
            "//tflite/kernels:builtin_ops",
            "//tflite/kernels:reference_ops",
        ],
        "//conditions:default": [
            ":litert_cpu_options",
            "//litert/runtime/accelerators/xnnpack:xnnpack_accelerator",  # buildcleaner: keep
            "//tflite/kernels:builtin_ops",
            "//tflite/kernels:reference_ops",
        ],
    }),
)

cc_test(
    name = "compiled_model_test",
    srcs = ["compiled_model_test.cc"],
    data = [
        "//litert/test:testdata/simple_add_dynamic_shape.tflite",
        "//litert/test:testdata/simple_model.tflite",
    ],
    linkopts = litert_android_linkopts(),
    # require GPU to run OpenCL tests.
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":compiled_model",
        ":open_cl_memory",
        ":tensor_buffer",
        ":tensor_buffer_requirements",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_layout",
        "//litert/c:litert_model",
        "//litert/c:litert_opaque_options_header",
        "//litert/c:litert_profiler",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_tensor_buffer_registry",
        "//litert/c/options:litert_cpu_options",
        "//litert/c/options:litert_runtime_options",
        "//litert/cc:litert_any",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_consts",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/options:litert_cpu_options",
        "//litert/cc/options:litert_runtime_options",
        "//litert/core:options",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "//third_party/odml/litert/ml_drift/delegate:buffer_handler_opencl",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/cl:cl_command_queue",
        "@ml_drift//ml_drift/cl:cl_context",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
    ],
)

cc_test(
    name = "gl_buffer_test",
    srcs = ["gl_buffer_test.cc"],
    linkopts = litert_android_linkopts(),
    tags = [
        "nofastbuild",
        "noios",
        "nolinux",
        "nomacos",
        "nowindows",
    ],
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":ahwb_buffer",
        ":gl_buffer",
        ":gpu_environment_header",
        ":tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_event",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_platform_support",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/test:matchers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/types:span",
    ] + gtest_main_no_heapcheck_deps() + select({
        "//litert:android": [
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

cc_library(
    name = "accelerator",
    hdrs = ["accelerator.h"],
    deps = [
        ":metrics",
        "//litert/c:litert_common",
    ],
)

cc_library(
    name = "accelerator_registry",
    srcs = ["accelerator_registry.cc"],
    hdrs = ["accelerator_registry.h"],
    deps = [
        ":accelerator",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/internal:litert_shared_library",
    ],
)

cc_library(
    name = "custom_op_dispatcher",
    srcs = ["custom_op_dispatcher.cc"],
    hdrs = ["custom_op_dispatcher.h"],
    copts = litert_metal_opts(),
    linkopts = litert_metal_linkopts(),
    deps = [
        ":external_litert_buffer_context",
        ":tensor_buffer",
        ":tfl_utils",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:options",
        "//tflite:builtin_ops",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "litert_runtime_options",
    srcs = ["litert_runtime_options.cc"],
    hdrs = ["litert_runtime_options.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:litert_toml_parser",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_runtime_options_test",
    srcs = ["litert_runtime_options_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":litert_runtime_options",
        "//litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
        "@tomlplusplus",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "litert_fabric_options",
#     hdrs = ["fabric/litert_fabric_options.h"],
# )
# copybara:uncomment_end

cc_library(
    name = "profiler_summarizer",
    srcs = ["profiler_summarizer.cc"],
    hdrs = ["profiler_summarizer.h"],
    deps = [
        "//tflite:framework",
        "//tflite/core:private_cc_api_stable",
        "//tflite/core/api",
        "//tflite/profiling:profile_buffer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        ":profiler_summarizer",
        "//litert/c:litert_profiler_event",
        "//tflite:framework",
        "//tflite/core:private_cc_api_stable",
        "//tflite/core/api",
        "//tflite/profiling:profile_buffer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "litert_cpu_options",
    srcs = ["litert_cpu_options.cc"],
    hdrs = ["litert_cpu_options.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_macros",
        "//litert/core:litert_toml_parser",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_cpu_options_test",
    srcs = ["litert_cpu_options_test.cc"],
    deps = [
        ":litert_cpu_options",
        "//litert/c:litert_common",
        "//litert/c/options:litert_cpu_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_gpu_util",
    hdrs = ["litert_gpu_util.h"],
    deps = [
        "//litert/c:litert_model_header",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "litert_gpu_util_test",
    srcs = ["litert_gpu_util_test.cc"],
    deps = [
        ":litert_gpu_util",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/test:matchers",
        "//tflite/delegates/gpu/common:data_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    deps = [
        ":profiler",
        "//litert/c:litert_profiler_event",
        "//litert/c/internal:litert_logging",
        "//tflite/core/api",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "profiler_summarizer_test",
    srcs = ["profiler_summarizer_test.cc"],
    deps = [
        ":profiler",
        ":profiler_summarizer",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/core/api",
        "//tflite/profiling:profile_buffer",
        "//tflite/profiling:profile_summarizer",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "metal_info",
    srcs = select({
        "//litert:ios": ["metal_info.cc"],
        "//litert:macos": ["metal_info.cc"],
        "//conditions:default": [],
    }),
    hdrs = ["metal_info.h"],
    copts = litert_metal_opts(),
    linkopts = litert_metal_linkopts(),
    deps = [
        "//litert/c:litert_common",
        "@com_google_absl//absl/log:absl_log",
    ] + litert_metal_deps_without_gpu_environment(),
    alwayslink = True,
)
