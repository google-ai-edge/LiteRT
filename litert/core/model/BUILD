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

load("//litert/build_common:litert_build_defs.bzl", "cc_library_with_testonly_vis")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert:litert_internal_users",
    ],
)

cc_library_with_testonly_vis(
    name = "model",
    srcs = ["model.cc"],
    hdrs = [
        "model.h",
        "//litert/c:litert_model_hdrs",
    ],
    vis = [
        "//litert:litert_internal_users",
        # TODO: This needs to be removed.
        "//litert/vendors/qualcomm/compiler:__pkg__",
    ],
    deps = [
        ":buffer_manager",
        ":ir_allocator",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/core:build_stamp",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/core/c:private_c_api_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:model_builder_base",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":buffer_manager",
        ":model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core:build_stamp",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:matchers",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_load",
    srcs = ["model_load.cc"],
    hdrs = ["model_load.h"],
    deps = [
        ":buffer_manager",
        ":flatbuffer_to_litert",
        ":model",
        ":model_graph",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:model_builder_base",
    ],
)

cc_test(
    name = "model_file_test",
    srcs = ["model_file_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
        # copybara:uncomment "//tflite/java/demo/app/src/main/assets:mobilenet_v1_1.0_224.tflite",
    ],
    deps = [
        ":buffer_manager",
        ":graph_validation",
        ":model",
        ":model_file_test_util",
        ":model_load",
        ":model_serialize",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/core:dispatch_op_schema",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:test_models",
        "//tflite/schema:schema_fbs_with_mutable",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
    ],
)

cc_library(
    name = "model_serialize",
    srcs = ["model_serialize.cc"],
    hdrs = ["model_serialize.h"],
    deps = [
        ":litert_to_flatbuffer",
        ":model",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:build_stamp",
        "//litert/core:dispatch_op_schema",
        "//litert/core:insert_order_map",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs_with_mutable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
    ],
)

cc_library(
    name = "flatbuffer_to_litert",
    srcs = ["flatbuffer_to_litert.cc"],
    hdrs = ["flatbuffer_to_litert.h"],
    deps = [
        ":model",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "flatbuffer_to_litert_test",
    srcs = ["flatbuffer_to_litert_test.cc"],
    deps = [
        ":flatbuffer_to_litert",
        "//litert/c:litert_model",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_to_flatbuffer",
    srcs = ["litert_to_flatbuffer.cc"],
    hdrs = ["litert_to_flatbuffer.h"],
    deps = [
        ":model",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_to_flatbuffer_test",
    srcs = ["litert_to_flatbuffer_test.cc"],
    deps = [
        ":litert_to_flatbuffer",
        ":model",
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_buffer",
    srcs = ["model_buffer.cc"],
    hdrs = ["model_buffer.h"],
    deps = [
        ":model",
        ":model_load",
        ":model_serialize",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/core:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "model_file_test_util",
    testonly = 1,
    srcs = ["model_file_test_util.cc"],
    hdrs = ["model_file_test_util.h"],
    deps = [
        ":flatbuffer_to_litert",
        ":model",
        "//litert/c:litert_logging",
        "//litert/cc:litert_detail",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ir_allocator",
    hdrs = ["ir_allocator.h"],
    deps = ["@com_google_absl//absl/types:span"],
)

cc_test(
    name = "ir_allocator_test",
    srcs = ["ir_allocator_test.cc"],
    deps = [
        ":ir_allocator",
        ":model",
        "//litert/c:litert_op_code",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_testonly_vis(
    name = "model_graph",
    srcs = ["model_graph.cc"],
    hdrs = [
        "model_graph.h",
    ],
    deps = [
        ":model",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_consts",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "graph_validation",
    srcs = ["graph_validation.cc"],
    hdrs = ["graph_validation.h"],
    deps = [
        ":model",
        ":model_graph",
        "//litert/c:litert_logging",
        "//litert/cc:litert_detail",
    ],
)

cc_library(
    name = "buffer_manager",
    hdrs = ["buffer_manager.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
    ],
)

cc_test(
    name = "model_graph_test",
    srcs = ["model_graph_test.cc"],
    deps = [
        ":graph_validation",
        ":ir_allocator",
        ":model",
        ":model_graph",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_buffer_test",
    srcs = ["model_buffer_test.cc"],
    deps = [
        ":model",
        ":model_buffer",
        ":model_load",
        "//litert/c:litert_op_code",
        "//litert/core:dispatch_op_schema",
        "//litert/test:common",
        "//litert/test:simple_cascade_model_npu",
        "//litert/test:simple_model_npu",
        "//tflite:framework",
        "//tflite:model_builder",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:allocation",
    ],
)

cc_test(
    name = "buffer_manager_test",
    srcs = ["buffer_manager_test.cc"],
    deps = [
        ":buffer_manager",
        "//litert/cc:litert_buffer_ref",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
