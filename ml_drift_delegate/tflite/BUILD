# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@ml_drift//ml_drift/common:build_config.bzl", "mld_platform_alias")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(["custom_parsers.h"])

cc_library(
    name = "ir_model_builder",
    srcs = ["ir_model_builder.cc"],
    hdrs = ["ir_model_builder.h"],
    deps = [
        ":ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common/transformations/ir:transform",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:shape",
        "//ml_drift_delegate/tflite/convert:convert_absolute_positional_embedding",
        "//ml_drift_delegate/tflite/convert:convert_argmax",
        "//ml_drift_delegate/tflite/convert:convert_batch_matmul",
        "//ml_drift_delegate/tflite/convert:convert_bitcast",
        "//ml_drift_delegate/tflite/convert:convert_broadcast_in_dim",
        "//ml_drift_delegate/tflite/convert:convert_cbrt",
        "//ml_drift_delegate/tflite/convert:convert_clamp",
        "//ml_drift_delegate/tflite/convert:convert_concat",
        "//ml_drift_delegate/tflite/convert:convert_conv",
        "//ml_drift_delegate/tflite/convert:convert_cumsum",
        "//ml_drift_delegate/tflite/convert:convert_depth_to_space",
        "//ml_drift_delegate/tflite/convert:convert_depthwise_conv",
        "//ml_drift_delegate/tflite/convert:convert_dequantize",
        "//ml_drift_delegate/tflite/convert:convert_dynamic_update_slice",
        "//ml_drift_delegate/tflite/convert:convert_elementwise",
        "//ml_drift_delegate/tflite/convert:convert_embedding_lookup",
        "//ml_drift_delegate/tflite/convert:convert_fully_connected",
        "//ml_drift_delegate/tflite/convert:convert_gather",
        "//ml_drift_delegate/tflite/convert:convert_group_norm",
        "//ml_drift_delegate/tflite/convert:convert_layer_norm",
        "//ml_drift_delegate/tflite/convert:convert_one_hot",
        "//ml_drift_delegate/tflite/convert:convert_pack",
        "//ml_drift_delegate/tflite/convert:convert_pad",
        "//ml_drift_delegate/tflite/convert:convert_pixel_shuffle",
        "//ml_drift_delegate/tflite/convert:convert_pooling2d",
        "//ml_drift_delegate/tflite/convert:convert_prelu",
        "//ml_drift_delegate/tflite/convert:convert_quantize",
        "//ml_drift_delegate/tflite/convert:convert_reduce",
        "//ml_drift_delegate/tflite/convert:convert_relu",
        "//ml_drift_delegate/tflite/convert:convert_resampler",
        "//ml_drift_delegate/tflite/convert:convert_reshape",
        "//ml_drift_delegate/tflite/convert:convert_resize2d",
        "//ml_drift_delegate/tflite/convert:convert_reverse",
        "//ml_drift_delegate/tflite/convert:convert_rms_norm",
        "//ml_drift_delegate/tflite/convert:convert_rope",
        "//ml_drift_delegate/tflite/convert:convert_sdpa",
        "//ml_drift_delegate/tflite/convert:convert_select",
        "//ml_drift_delegate/tflite/convert:convert_slice",
        "//ml_drift_delegate/tflite/convert:convert_softmax",
        "//ml_drift_delegate/tflite/convert:convert_space_to_depth",
        "//ml_drift_delegate/tflite/convert:convert_split",
        "//ml_drift_delegate/tflite/convert:convert_strided_slice",
        "//ml_drift_delegate/tflite/convert:convert_tile",
        "//ml_drift_delegate/tflite/convert:convert_topk",
        "//ml_drift_delegate/tflite/convert:convert_transpose",
        "//ml_drift_delegate/tflite/convert:convert_transpose_conv",
        "//ml_drift_delegate/tflite/convert:convert_unpack",
        "//ml_drift_delegate/tflite/convert:convert_unpooling2d",
        # copybara:uncomment_begin(google-only)
        # "//ml_drift_delegate/tflite/convert/google:convert_alignment_points_to_transform_matrix",
        # "//ml_drift_delegate/tflite/convert/google:convert_keep_if_max_2d_pt2",
        # "//ml_drift_delegate/tflite/convert/google:convert_landmarks_to_transform_matrix",
        # "//ml_drift_delegate/tflite/convert/google:convert_roi_to_transform_matrix",
        # "//ml_drift_delegate/tflite/convert/google:convert_transform_landmarks",
        # "//ml_drift_delegate/tflite/convert/google:convert_transform_tensor_bilinear",
        # copybara:uncomment_end
        "//ml_drift_delegate/tflite/support",
        ":custom_ir_operation_parser",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:framework_stable",
        "//tflite:model_builder",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
    ],
)

cc_library(
    name = "ir_model_builder_helper",
    srcs = ["ir_model_builder_helper.cc"],
    hdrs = ["ir_model_builder_helper.h"],
    deps = [
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/kernels/internal:tensor_utils_no_eigen",
        "//tflite/kernels/internal:types",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "ir_model_builder_helper_test",
    srcs = ["ir_model_builder_helper_test.cc"],
    deps = [
        ":ir_model_builder_helper",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "custom_ir_operation_parser",
    hdrs = ["custom_ir_operation_parser.h"],
    deps = [
        ":ir_model_builder_helper",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_library(
    name = "lstm_parser",
    srcs = ["lstm_parser.cc"],
    hdrs = ["lstm_parser.h"],
    deps = [
        ":model_builder_helper",
        ":object_reader",
        "//tflite/c:common",
        "//tflite/kernels:lstm_shared",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "object_reader",
    srcs = ["object_reader.cc"],
    hdrs = ["object_reader.h"],
    deps = [
        ":model_builder_helper",
        ":shared_const_tensor_map",
        "//tflite/c:common",
        "//tflite/delegates:utils",
        "//tflite/kernels:kernel_util",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "object_reader_test",
    srcs = ["object_reader_test.cc"],
    deps = [
        ":object_reader",
        ":shared_const_tensor_map",
        ":stub_tflite_context",
        "//tflite:builtin_ops",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/testing:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
    ],
)

cc_library(
    name = "operation_parser",
    srcs = ["operation_parser.cc"],
    hdrs = ["operation_parser.h"],
    deps = [
        ":model_builder_helper",
        ":object_reader",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//tflite/tools/versioning:gpu_compatibility",
        "//tflite/tools/versioning:op_signature",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "operation_parser_test",
    srcs = ["operation_parser_test.cc"],
    deps = [
        ":operation_parser",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/testing:matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
    ],
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    deps = [
        "//tflite/c:common",
        "//tflite/kernels/internal:optimized_base",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "model_builder_internal",
    hdrs = ["model_builder_internal.h"],
    deps = [
        ":operation_parser",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "model_builder_helper",
    srcs = ["model_builder_helper.cc"],
    hdrs = ["model_builder_helper.h"],
    deps = [
        "//tflite:kernel_api",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/kernels/internal:tensor_utils_no_eigen",
        "//tflite/kernels/internal:types",
        "@FP16",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "model_builder",
    srcs = ["model_builder.cc"],
    hdrs = [
        "model_builder.h",
        "model_builder_internal.h",
    ],
    deps = [
        ":lstm_parser",
        ":model_builder_helper",
        ":object_reader",
        ":operation_parser",
        ":shared_const_tensor_map",
        "//tflite:framework",
        "//tflite:kernel_api",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/c:private_c_api_without_op_resolver",
        "//tflite/core/c:private_common",
        "//tflite/delegates:utils",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:lstm_shared",
        "//tflite/kernels/internal:optimized_base",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:tensor_utils_no_eigen",
        "//tflite/tools/versioning:gpu_compatibility",
        "//tflite/tools/versioning:op_signature",
        "@FP16",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/kernels:dynamic_update_slice",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/transformations:model_transformations",
    ] + mld_platform_alias("custom_parsers", "@ml_drift//ml_drift/common/"),
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "model_builder_tflite_model_test",
#     srcs = ["model_builder_tflite_model_test.cc"],
#     deps = [
#         ":model_builder",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/strings",
#         "@ml_drift//ml_drift/common:model",
#         "@ml_drift//ml_drift/common:operations",
#         "@ml_drift//ml_drift/common:shape",
#         "@ml_drift//ml_drift/common:status",
#         "//ml_drift_delegate/tflite/model_builder_tflite_models:embedded_models",
#         "//tflite:framework_stable",
#         "//tflite/kernels:builtin_ops",
#     ],
# )
# copybara:uncomment_end

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    deps = [
        ":quantization_util",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_test(
    name = "model_builder_test",
    srcs = ["model_builder_test.cc"],
    deps = [
        ":model_builder",
        ":object_reader",
        ":operation_parser",
        ":shared_const_tensor_map",
        ":stub_tflite_context",
        "//tflite:framework",
        "//tflite:kernel_api",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_test(
    name = "model_builder_helper_test",
    srcs = ["model_builder_helper_test.cc"],
    deps = [
        ":model_builder_helper",
        "//tflite/core/c:private_common",
        "//tflite/testing:matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shared_const_tensor_map",
    hdrs = ["shared_const_tensor_map.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "stub_tflite_context",
    testonly = True,
    hdrs = ["stub_tflite_context.h"],
    deps = [
        "//tflite:builtin_ops",
        "//tflite:util",
        "//tflite/c:common",
        "@com_google_absl//absl/log:absl_check",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "unimplemented_operation_parser",
    hdrs = ["unimplemented_operation_parser.h"],
    deps = [
        ":object_reader",
        ":operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "testing_utils",
    testonly = True,
    srcs = ["testing_utils.cc"],
    hdrs = ["testing_utils.h"],
    deps = [
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/status",
        "@ml_drift//ml_drift/common/task:testing_util",
    ],
)

cc_library(
    name = "sample_tflite_loader",
    srcs = ["sample_tflite_loader.cc"],
    hdrs = ["sample_tflite_loader.h"],
    deps = [
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/samples/stable_diffusion:model_data_loader",
    ],
)
