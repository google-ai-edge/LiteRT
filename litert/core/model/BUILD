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

load("//litert/build_common:special_rule.bzl", "lite_rt_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        "//litert:__subpackages__",
    ] + lite_rt_friends(),
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = [
        "model.h",
        "//litert/c:litert_model_hdrs",
    ],
    deps = [
        ":buffer_manager",
        ":ir_allocator",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/core/util:flatbuffer_tools",
        "//third_party/tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//third_party/tensorflow/lite/core/c:c_api_types",
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
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:matchers",
        "//third_party/tensorflow/lite/schema:schema_fbs",
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
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/util:flatbuffer_tools",
        "//third_party/tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//third_party/tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "model_file_test",
    srcs = ["model_file_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
        # copybara:uncomment "//third_party/tensorflow/lite/java/demo/app/src/main/assets:mobilenet_v1_1.0_224.tflite",
    ],
    deps = [
        ":buffer_manager",
        ":graph_validation",
        ":model",
        ":model_file_test_util",
        ":model_load",
        ":model_serialize",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
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
        "//third_party/tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//third_party/tensorflow/lite/schema:schema_fbs",
        "//third_party/tensorflow/lite/schema:schema_fbs_with_mutable",
    ],
)

cc_library(
    name = "model_serialize",
    srcs = ["model_serialize.cc"],
    hdrs = ["model_serialize.h"],
    deps = [
        ":litert_to_flatbuffer",
        ":model",
        "//third_party/absl/container:flat_hash_map",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:build_stamp",
        "//litert/core:dispatch_op_schema",
        "//litert/core:insert_order_map",
        "//litert/core/util:flatbuffer_tools",
        "//third_party/tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//third_party/tensorflow/lite/schema:schema_fbs",
        "//third_party/tensorflow/lite/schema:schema_fbs_with_mutable",
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
        "//third_party/tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "flatbuffer_to_litert_test",
    srcs = ["flatbuffer_to_litert_test.cc"],
    deps = [
        ":flatbuffer_to_litert",
        "//testing/base/public:gunit_main",
        "//third_party/absl/types:span",
        "//litert/c:litert_model",
        "//litert/core/util:flatbuffer_tools",
    ],
)

cc_library(
    name = "litert_to_flatbuffer",
    srcs = ["litert_to_flatbuffer.cc"],
    hdrs = ["litert_to_flatbuffer.h"],
    deps = [
        ":model",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/core/util:flatbuffer_tools",
        "//third_party/tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "litert_to_flatbuffer_test",
    srcs = ["litert_to_flatbuffer_test.cc"],
    deps = [
        ":litert_to_flatbuffer",
        ":model",
        "//testing/base/public:gunit_main",
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
        "//litert/core/util:flatbuffer_tools",
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
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/core:filesystem",
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
        "//third_party/absl/types:span",
        "//litert/c:litert_logging",
        "//litert/cc:litert_detail",
        "//litert/core/util:flatbuffer_tools",
    ],
)

cc_library(
    name = "ir_allocator",
    hdrs = ["ir_allocator.h"],
    deps = ["//third_party/absl/types:span"],
)

cc_test(
    name = "ir_allocator_test",
    srcs = ["ir_allocator_test.cc"],
    deps = [
        ":ir_allocator",
        ":model",
        "//testing/base/public:gunit_main",
        "//litert/c:litert_op_code",
    ],
)

cc_library(
    name = "model_graph",
    srcs = ["model_graph.cc"],
    hdrs = [
        "model_graph.h",
        "//litert/cc:litert_consts.h",
    ],
    deps = [
        ":model",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/log:absl_check",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
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
        ":model",
        ":model_graph",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
    ],
)

cc_test(
    name = "model_buffer_test",
    srcs = ["model_buffer_test.cc"],
    deps = [
        ":model",
        ":model_buffer",
        ":model_load",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_op_code",
        "//litert/core:dispatch_op_schema",
        "//litert/test:common",
        "//litert/test:simple_cascade_model_npu",
        "//litert/test:simple_model_npu",
        "//third_party/tensorflow/compiler/mlir/lite:allocation",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:model_builder",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/core:cc_api_stable",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "buffer_manager_test",
    srcs = ["buffer_manager_test.cc"],
    deps = [
        ":buffer_manager",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//litert/cc:litert_buffer_ref",
    ],
)
