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

cc_library(
    name = "stub_context",
    testonly = True,
    srcs = ["stub_context.cc"],
    hdrs = ["stub_context.h"],
    deps = [
        "//tflite:builtin_ops",
        "//tflite:util",
        "//tflite/c:common",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "support",
    srcs = ["support.cc"],
    hdrs = ["support.h"],
    deps = [
        ":support_absolute_positional_embedding",
        ":support_argmax",
        ":support_aux",
        ":support_batch_matmul",
        ":support_bitcast",
        ":support_broadcast_in_dim",
        ":support_clamp",
        ":support_concat",
        ":support_conv",
        ":support_cumsum",
        ":support_depth_to_space",
        ":support_depthwise_conv",
        ":support_dequantize",
        ":support_dynamic_update_slice",
        ":support_elementwise",
        ":support_embedding_lookup",
        ":support_fully_connected",
        ":support_gather",
        ":support_group_norm",
        ":support_layer_norm",
        ":support_one_hot",
        ":support_pack",
        ":support_pad",
        ":support_pixel_shuffle",
        ":support_pooling2d",
        ":support_prelu",
        ":support_quantize",
        ":support_reduce",
        ":support_relu",
        ":support_resampler",
        ":support_reshape",
        ":support_resize2d",
        ":support_reverse",
        ":support_rms_norm",
        ":support_rotary_positional_embedding",
        ":support_sdpa",
        ":support_select",
        ":support_slice",
        ":support_softmax",
        ":support_space_to_depth",
        ":support_split",
        ":support_splitv",
        ":support_strided_slice",
        ":support_tile",
        ":support_topk",
        ":support_transpose",
        ":support_transpose_conv",
        ":support_unpack",
        ":support_unpooling2d",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        # copybara:uncomment_begin(google-only)
        # "//third_party/odml/litert/ml_drift/tflite/support/google:support_alignment_points_to_transform_matrix",
        # "//third_party/odml/litert/ml_drift/tflite/support/google:support_keep_if_max_2d_pt2",
        # "//third_party/odml/litert/ml_drift/tflite/support/google:support_landmarks_to_transform_matrix",
        # "//third_party/odml/litert/ml_drift/tflite/support/google:support_roi_to_transform_matrix",
        # "//third_party/odml/litert/ml_drift/tflite/support/google:support_transform_landmarks",
        # "//third_party/odml/litert/ml_drift/tflite/support/google:support_transform_tensor_bilinear",
        # copybara:uncomment_end
        "//tflite:builtin_ops",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/delegates:utils",
        "//tflite/kernels:kernel_util",
        "//tflite/schema:schema_fbs",
    ] + mld_platform_alias("custom_parsers", "@ml_drift//ml_drift/common/"),
)

cc_library(
    name = "support_aux",
    srcs = ["support_aux.cc"],
    hdrs = ["support_aux.h"],
    deps = [
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:operation_parser",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_test(
    name = "support_aux_test",
    srcs = ["support_aux_test.cc"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_absolute_positional_embedding",
    srcs = ["support_absolute_positional_embedding.cc"],
    hdrs = ["support_absolute_positional_embedding.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_argmax",
    srcs = ["support_argmax.cc"],
    hdrs = ["support_argmax.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_batch_matmul",
    srcs = ["support_batch_matmul.cc"],
    hdrs = ["support_batch_matmul.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_bitcast",
    srcs = ["support_bitcast.cc"],
    hdrs = ["support_bitcast.h"],
    deps = [
        ":support_aux",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_broadcast_in_dim",
    srcs = ["support_broadcast_in_dim.cc"],
    hdrs = ["support_broadcast_in_dim.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_clamp",
    srcs = ["support_clamp.cc"],
    hdrs = ["support_clamp.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_concat",
    srcs = ["support_concat.cc"],
    hdrs = ["support_concat.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "support_conv",
    srcs = ["support_conv.cc"],
    hdrs = ["support_conv.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_cumsum",
    srcs = ["support_cumsum.cc"],
    hdrs = ["support_cumsum.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_depth_to_space",
    srcs = ["support_depth_to_space.cc"],
    hdrs = ["support_depth_to_space.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_depthwise_conv",
    srcs = ["support_depthwise_conv.cc"],
    hdrs = ["support_depthwise_conv.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_dequantize",
    srcs = ["support_dequantize.cc"],
    hdrs = ["support_dequantize.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_elementwise",
    srcs = ["support_elementwise.cc"],
    hdrs = ["support_elementwise.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_embedding_lookup",
    srcs = ["support_embedding_lookup.cc"],
    hdrs = ["support_embedding_lookup.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_fully_connected",
    srcs = ["support_fully_connected.cc"],
    hdrs = ["support_fully_connected.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_gather",
    srcs = ["support_gather.cc"],
    hdrs = ["support_gather.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_group_norm",
    srcs = ["support_group_norm.cc"],
    hdrs = ["support_group_norm.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_layer_norm",
    srcs = ["support_layer_norm.cc"],
    hdrs = ["support_layer_norm.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_one_hot",
    srcs = ["support_one_hot.cc"],
    hdrs = ["support_one_hot.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "support_pack",
    srcs = ["support_pack.cc"],
    hdrs = ["support_pack.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_pad",
    srcs = ["support_pad.cc"],
    hdrs = ["support_pad.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_pixel_shuffle",
    srcs = ["support_pixel_shuffle.cc"],
    hdrs = ["support_pixel_shuffle.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "support_pooling2d",
    srcs = ["support_pooling2d.cc"],
    hdrs = ["support_pooling2d.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_prelu",
    srcs = ["support_prelu.cc"],
    hdrs = ["support_prelu.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_reshape",
    srcs = ["support_reshape.cc"],
    hdrs = ["support_reshape.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_tile",
    srcs = ["support_tile.cc"],
    hdrs = ["support_tile.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "support_unpack",
    srcs = ["support_unpack.cc"],
    hdrs = ["support_unpack.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_resize2d",
    srcs = ["support_resize2d.cc"],
    hdrs = ["support_resize2d.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_reverse",
    srcs = ["support_reverse.cc"],
    hdrs = ["support_reverse.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_rms_norm",
    srcs = ["support_rms_norm.cc"],
    hdrs = ["support_rms_norm.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_sdpa",
    srcs = ["support_sdpa.cc"],
    hdrs = ["support_sdpa.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "support_select",
    srcs = ["support_select.cc"],
    hdrs = ["support_select.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_split",
    srcs = ["support_split.cc"],
    hdrs = ["support_split.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_space_to_depth",
    srcs = ["support_space_to_depth.cc"],
    hdrs = ["support_space_to_depth.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_splitv",
    srcs = ["support_splitv.cc"],
    hdrs = ["support_splitv.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_softmax",
    srcs = ["support_softmax.cc"],
    hdrs = ["support_softmax.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_reduce",
    srcs = ["support_reduce.cc"],
    hdrs = ["support_reduce.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_relu",
    srcs = ["support_relu.cc"],
    hdrs = ["support_relu.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_lstm",
    srcs = ["support_lstm.cc"],
    hdrs = ["support_lstm.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:lstm_shared",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_resampler",
    srcs = ["support_resampler.cc"],
    hdrs = ["support_resampler.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_rotary_positional_embedding",
    srcs = ["support_rotary_positional_embedding.cc"],
    hdrs = ["support_rotary_positional_embedding.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_strided_slice",
    srcs = ["support_strided_slice.cc"],
    hdrs = ["support_strided_slice.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:util",
    ],
)

cc_library(
    name = "support_dynamic_update_slice",
    srcs = ["support_dynamic_update_slice.cc"],
    hdrs = ["support_dynamic_update_slice.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_slice",
    srcs = ["support_slice.cc"],
    hdrs = ["support_slice.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_transpose",
    srcs = ["support_transpose.cc"],
    hdrs = ["support_transpose.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_transpose_conv",
    srcs = ["support_transpose_conv.cc"],
    hdrs = ["support_transpose_conv.h"],
    deps = [
        ":support_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "support_unpooling2d",
    srcs = ["support_unpooling2d.cc"],
    hdrs = ["support_unpooling2d.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_topk",
    srcs = ["support_topk.cc"],
    hdrs = ["support_topk.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "support_quantize",
    srcs = ["support_quantize.cc"],
    hdrs = ["support_quantize.h"],
    deps = [
        ":support_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/base:nullability",
    ],
)

cc_test(
    name = "support_absolute_positional_embedding_test",
    srcs = ["support_absolute_positional_embedding_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_argmax_test",
    srcs = ["support_argmax_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_batch_matmul_test",
    srcs = ["support_batch_matmul_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_bitcast_test",
    srcs = ["support_bitcast_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_binary_arithmetic_test",
    srcs = ["support_binary_arithmetic_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_binary_logical_test",
    srcs = ["support_binary_logical_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_broadcast_in_dim_test",
    srcs = ["support_broadcast_in_dim_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_clamp_test",
    srcs = ["support_clamp_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_concat_test",
    srcs = ["support_concat_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_conv_test",
    srcs = ["support_conv_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_cumsum_test",
    srcs = ["support_cumsum_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_depth_to_space_test",
    srcs = ["support_depth_to_space_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_depthwise_conv_test",
    srcs = ["support_depthwise_conv_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_dequantize_test",
    srcs = ["support_dequantize_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_embedding_lookup_test",
    srcs = ["support_embedding_lookup_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_fully_connected_test",
    srcs = ["support_fully_connected_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_gather_test",
    srcs = ["support_gather_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_group_norm_test",
    srcs = ["support_group_norm_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "support_layer_norm_test",
    srcs = ["support_layer_norm_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

# TODO(b/438520776): Re-enable once LSTM convert is live.
# cc_test(
#     name = "support_lstm_test",
#     srcs = ["support_lstm_test.cc"],
#     deps = [
#         ":stub_context",
#         ":support",  # buildcleaner: keep
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/strings",
#         "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
#         "//tflite:builtin_ops",
#         "//tflite/c:common",
#         "//tflite/core/c:private_common",
#         "//tflite/kernels:lstm_shared",
#         "//tflite/schema:schema_fbs",
#     ],
# )

cc_test(
    name = "support_prelu_test",
    srcs = ["support_prelu_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_reduce_test",
    srcs = ["support_reduce_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_one_hot_test",
    srcs = ["support_one_hot_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_pack_test",
    srcs = ["support_pack_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_pad_test",
    srcs = ["support_pad_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_pixel_shuffle_test",
    srcs = ["support_pixel_shuffle_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "support_relu_test",
    srcs = ["support_relu_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_pooling2d_test",
    srcs = ["support_pooling2d_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_resampler_test",
    srcs = ["support_resampler_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_reshape_test",
    srcs = ["support_reshape_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_tile_test",
    srcs = ["support_tile_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_resize2d_test",
    srcs = ["support_resize2d_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_reverse_test",
    srcs = ["support_reverse_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_rotary_positional_embedding_test",
    srcs = ["support_rotary_positional_embedding_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_rms_norm_test",
    srcs = ["support_rms_norm_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "support_sdpa_test",
    srcs = ["support_sdpa_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_select_test",
    srcs = ["support_select_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_slice_test",
    srcs = ["support_slice_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_softmax_test",
    srcs = ["support_softmax_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_split_test",
    srcs = ["support_split_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_splitv_test",
    srcs = ["support_splitv_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_space_to_depth_test",
    srcs = ["support_space_to_depth_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_strided_slice_test",
    srcs = ["support_strided_slice_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_dynamic_update_slice_test",
    srcs = ["support_dynamic_update_slice_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_transpose_test",
    srcs = ["support_transpose_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_unary_arithmetic_test",
    srcs = ["support_unary_arithmetic_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_unary_logical_test",
    srcs = ["support_unary_logical_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/schema:schema_fbs",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_transpose_conv_builtin_test",
    srcs = ["support_transpose_conv_builtin_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_transpose_conv_custom_test",
    srcs = ["support_transpose_conv_custom_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_unpooling2d_test",
    srcs = ["support_unpooling2d_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_test",
    srcs = ["support_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_unpack_test",
    srcs = ["support_unpack_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_topk_test",
    srcs = ["support_topk_test.cc"],
    deps = [
        ":stub_context",
        ":support",  # buildcleaner: keep
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "support_quantize_test",
    srcs = ["support_quantize_test.cc"],
    deps = [
        ":support_quantize",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)
