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
#
# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_lib", "litert_test", "make_rpaths")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

litert_dynamic_lib(
    name = "qnn_compiler_plugin",
    srcs = [
        "qnn_compiler_plugin.cc",
    ],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
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
        "-Wl,-soname=libLiteRtCompilerPlugin_Qualcomm.so",
    ],
    shared_lib_name = "qnn_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Qualcomm.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        ":qnn_compose_graph",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_qualcomm_options",
        "//litert/vendors/cc:options_helper",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "qnn_compiler_plugin_test",
    srcs = [
        "qnn_compiler_plugin_test.cc",
    ],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "nosan",
    ],
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":qnn_compiler_plugin",  # buildcleaner: keep
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_qualcomm_options",
        "//litert/core/model:model_testonly",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "//litert/vendors/qualcomm:qnn_manager",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_exec(
    name = "qnn_compiler_plugin_device_test",
    backend_id = "qualcomm",
    data = ["//litert/test:testdata"],
    target = ":qnn_compiler_plugin_test",
)

litert_lib(
    name = "qnn_compose_graph",
    srcs = ["qnn_compose_graph.cc"],
    hdrs = ["qnn_compose_graph.h"],
    deps = [
        ":graph_mapper",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_op_options",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:broadcast_to_op_builder",
        "//litert/vendors/qualcomm/core/builders:cast_op_builder",
        "//litert/vendors/qualcomm/core/builders:concatenation_op_builder",
        "//litert/vendors/qualcomm/core/builders:conv2d_op_builder",
        "//litert/vendors/qualcomm/core/builders:cumsum_op_builder",
        "//litert/vendors/qualcomm/core/builders:depthwise_conv2d_op_builder",
        "//litert/vendors/qualcomm/core/builders:dynamic_update_slice_op_builder",
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//litert/vendors/qualcomm/core/builders:embedding_lookup_op_builder",
        "//litert/vendors/qualcomm/core/builders:fully_connected_op_builder",
        "//litert/vendors/qualcomm/core/builders:fully_connected_op_builder_htp",
        "//litert/vendors/qualcomm/core/builders:gather_op_builder",
        "//litert/vendors/qualcomm/core/builders:gathernd_op_builder",
        "//litert/vendors/qualcomm/core/builders:gelu_op_builder",
        "//litert/vendors/qualcomm/core/builders:hard_swish_op_builder",
        "//litert/vendors/qualcomm/core/builders:leaky_relu_op_builder",
        "//litert/vendors/qualcomm/core/builders:logistic_op_builder",
        "//litert/vendors/qualcomm/core/builders:matmul_op_builder",
        "//litert/vendors/qualcomm/core/builders:mean_op_builder",
        "//litert/vendors/qualcomm/core/builders:op_builder",
        "//litert/vendors/qualcomm/core/builders:pack_op_builder",
        "//litert/vendors/qualcomm/core/builders:pad_op_builder",
        "//litert/vendors/qualcomm/core/builders:pool2d_op_builder",
        "//litert/vendors/qualcomm/core/builders:quantize_op_builder",
        "//litert/vendors/qualcomm/core/builders:reduce_op_builder",
        "//litert/vendors/qualcomm/core/builders:relu6_op_builder",
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/builders:reshape_op_builder",
        "//litert/vendors/qualcomm/core/builders:resize_op_builder",
        "//litert/vendors/qualcomm/core/builders:rms_norm_op_builder",
        "//litert/vendors/qualcomm/core/builders:select_op_builder",
        "//litert/vendors/qualcomm/core/builders:slice_op_builder",
        "//litert/vendors/qualcomm/core/builders:softmax_op_builder",
        "//litert/vendors/qualcomm/core/builders:spatial_transform_op_builder",
        "//litert/vendors/qualcomm/core/builders:split_op_builder",
        "//litert/vendors/qualcomm/core/builders:tanh_op_builder",
        "//litert/vendors/qualcomm/core/builders:transpose_conv_op_builder",
        "//litert/vendors/qualcomm/core/builders:transpose_op_builder",
        "//litert/vendors/qualcomm/core/dump:dump_graph",
        "//litert/vendors/qualcomm/core/transformation:graph_to_graph",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_lib(
    name = "graph_mapper",
    srcs = [
        "graph_mapper.cc",
    ],
    hdrs = ["graph_mapper.h"],
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm:qnn_manager",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@qairt//:qnn_lib_headers",
    ],
)
