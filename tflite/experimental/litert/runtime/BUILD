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

load("//tflite/experimental/litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps")
load("//tflite/experimental/litert/build_common:special_rule.bzl", "gles_deps", "gles_linkopts", "lite_rt_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "event",
    srcs = [
        "event.cc",
    ],
    hdrs = [
        "event.h",
    ],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_expected",
    ],
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
        "open_cl_buffer.cc",
        "tensor_buffer.cc",
    ],
    hdrs = [
        "ahwb_buffer.h",
        "dmabuf_buffer.h",
        "event.h",
        "fastrpc_buffer.h",
        "gl_buffer.h",
        "gl_texture.h",
        "ion_buffer.h",
        "open_cl_buffer.h",
        "tensor_buffer.h",
        "tensor_buffer_requirements.h",
        "//tflite/experimental/litert/c:litert_event.h",
        "//tflite/experimental/litert/c:litert_tensor_buffer.h",
        "//tflite/experimental/litert/c:litert_tensor_buffer_requirements.h",
    ],
    linkopts = gles_linkopts(),
    deps = [
        ":environment",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_layout",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer_types",
        "//tflite/experimental/litert/cc:litert_event",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/cc:litert_tensor_buffer_utils",
        "//tflite/experimental/litert/core/util:tensor_type_util",
        "//tflite/experimental/litert/runtime/opencl:buffer",
        "//tflite/experimental/litert/runtime/opencl:cl_command_queue",
        "//tflite/experimental/litert/runtime/opencl:cl_context",
        "//tflite/experimental/litert/runtime/opencl:opencl_wrapper",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_deps() + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:gl_buffer",
            "//tflite/delegates/gpu/gl:gl_texture",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "environment",
    srcs = [
        "environment.cc",
    ],
    hdrs = [
        "environment.h",
    ],
    visibility = [
        "@org_tensorflow//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        "//tflite/experimental/litert/c:litert_any",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/core:environment",
        "//tflite/experimental/litert/runtime/opencl:cl_command_queue",
        "//tflite/experimental/litert/runtime/opencl:cl_context",
        "//tflite/experimental/litert/runtime/opencl:cl_device",
        "//tflite/experimental/litert/runtime/opencl:opencl_wrapper",
        "@opencl_headers",
    ],
)

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":environment",
        "@com_google_googletest//:gtest_main",
        # copybara:uncomment_begin(google-only)
        # "@org_tensorflow//third_party/ml_drift/cl:environment",
        # "@org_tensorflow//third_party/ml_drift/cl:opencl_wrapper",
        # copybara:uncomment_end
        "//tflite/experimental/litert/c:litert_environment",
        "//tflite/experimental/litert/cc:litert_any",
        "//tflite/experimental/litert/runtime/opencl:opencl_wrapper",
    ],
)

cc_library(
    name = "tfl_utils",
    srcs = [
        "tfl_utils.cc",
    ],
    hdrs = [
        "tfl_utils.h",
    ],
    deps = [
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_detail",
        "//tflite/experimental/litert/cc:litert_element_type",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_model",
    ],
)

cc_library(
    name = "external_litert_buffer_context",
    srcs = ["external_litert_buffer_context.cc"],
    hdrs = ["external_litert_buffer_context.h"],
    visibility = [
        "//tflite/experimental/litert:__subpackages__",
    ] + lite_rt_friends(),
    deps = [
        ":tfl_utils",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_tensor_buffer_requirements",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "compiled_model",
    srcs = ["compiled_model.cc"],
    hdrs = ["compiled_model.h"],
    deps = [
        ":compilation_options",
        ":external_litert_buffer_context",
        ":tensor_buffer",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:model_builder",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/delegates/utils:simple_opaque_delegate",
        "//tflite/experimental/litert/c:litert_accelerator_options",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_compilation_options",
        "//tflite/experimental/litert/c:litert_dispatch_delegate",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_event",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_tensor_buffer_requirements",
        "//tflite/experimental/litert/compiler/plugin:compiler_plugin",
        "//tflite/experimental/litert/core:accelerator",
        "//tflite/experimental/litert/core:accelerator_model_compilation_data",
        "//tflite/experimental/litert/core:build_stamp",
        "//tflite/experimental/litert/core:environment",
        "//tflite/experimental/litert/core/model",
        "//tflite/experimental/litert/runtime/dispatch:dispatch_delegate",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:allocation",
    ],
)

cc_test(
    name = "compiled_model_test",
    srcs = ["compiled_model_test.cc"],
    data = [
        "//tflite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":compiled_model",
        ":tensor_buffer",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_compilation_options",
        "//tflite/experimental/litert/c:litert_environment",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/core/model",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:matchers",
        "//tflite/experimental/litert/test:simple_model",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "compilation_options",
    hdrs = [
        "compilation_options.h",
        "//tflite/experimental/litert/c:litert_compilation_options.h",
    ],
    deps = [
        "//tflite/experimental/litert/c:litert_accelerator_options",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_layout",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
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
    tags = [
        "notap",
    ],
    deps = [
        ":tensor_buffer",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/test:matchers",
    ] + gtest_main_no_heapcheck_deps() + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:egl_environment",
            "//tflite/delegates/gpu/gl:gl_buffer",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensor_buffer_conversion",
    srcs = ["tensor_buffer_conversion.cc"],
    hdrs = ["tensor_buffer_conversion.h"],
    linkopts = gles_linkopts(),
    deps = [
        ":tensor_buffer",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_environment",
        "//tflite/experimental/litert/c:litert_tensor_buffer_types",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/cc:litert_tensor_buffer_utils",
        "@com_google_absl//absl/strings:str_format",
    ] + gles_deps(),
)

cc_test(
    name = "tensor_buffer_conversion_test",
    srcs = ["tensor_buffer_conversion_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "notap",
    ],
    deps = [
        ":tensor_buffer",
        ":tensor_buffer_conversion",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_environment",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer_types",
        "//tflite/experimental/litert/cc:litert_layout",
        "//tflite/experimental/litert/core:environment",
        "//tflite/experimental/litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)
