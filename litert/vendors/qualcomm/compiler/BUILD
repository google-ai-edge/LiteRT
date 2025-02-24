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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_lib", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "qnn_compiler_plugin",
    srcs = ["qnn_compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    shared_lib_name = "qnn_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Qualcomm.so",
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    ungrte = True,
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":qnn_compose_graph",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/core/model",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

litert_test(
    name = "qnn_compiler_plugin_test",
    srcs = [
        "qnn_compiler_plugin_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "//third_party/bazel_platforms/os:android": [],
        "//third_party/bazel_platforms/os:linux": [],
        "//conditions:default": ["//third_party/bazel_platforms:incompatible"],
    }),
    use_sys_malloc = True,
    deps = [
        ":qnn_compiler_plugin",  # buildcleaner: keep
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//litert/vendors/qualcomm/compiler/IR:qnn_tensor",
        "//litert/vendors/qualcomm/compiler/legalizations:quantize_op_legalization",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

litert_lib(
    name = "qnn_compose_graph",
    srcs = ["qnn_compose_graph.cc"],
    hdrs = ["qnn_compose_graph.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":graph_mapper",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/c:litert_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/core/model",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:cast_op_builder",
        "//litert/vendors/qualcomm/core/builders:concatenation_op_builder",
        "//litert/vendors/qualcomm/core/builders:dynamic_update_slice_op_builder",
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//litert/vendors/qualcomm/core/builders:embedding_lookup_op_builder",
        "//litert/vendors/qualcomm/core/builders:fully_connected_op_builder",
        "//litert/vendors/qualcomm/core/builders:gather_op_builder",
        "//litert/vendors/qualcomm/core/builders:gelu_op_builder",
        "//litert/vendors/qualcomm/core/builders:matmul_op_builder",
        "//litert/vendors/qualcomm/core/builders:mean_op_builder",
        "//litert/vendors/qualcomm/core/builders:pack_op_builder",
        "//litert/vendors/qualcomm/core/builders:quantize_op_builder",
        "//litert/vendors/qualcomm/core/builders:reduce_op_builder",
        "//litert/vendors/qualcomm/core/builders:reshape_op_builder",
        "//litert/vendors/qualcomm/core/builders:select_op_builder",
        "//litert/vendors/qualcomm/core/builders:slice_op_builder",
        "//litert/vendors/qualcomm/core/builders:softmax_op_builder",
        "//litert/vendors/qualcomm/core/builders:split_op_builder",
        "//litert/vendors/qualcomm/core/builders:tanh_op_builder",
        "//litert/vendors/qualcomm/core/builders:transpose_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

litert_lib(
    name = "graph_mapper",
    srcs = [
        "graph_mapper.cc",
    ],
    hdrs = ["graph_mapper.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//litert/vendors/qualcomm/compiler/IR:qnn_tensor",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)
