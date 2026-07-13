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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "convert_aux",
    srcs = ["convert_aux.cc"],
    hdrs = ["convert_aux.h"],
    deps = [
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor_ctypes",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "convert_absolute_positional_embedding",
    srcs = ["convert_absolute_positional_embedding.cc"],
    hdrs = ["convert_absolute_positional_embedding.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_batch_matmul",
    srcs = ["convert_batch_matmul.cc"],
    hdrs = ["convert_batch_matmul.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_broadcast_in_dim",
    srcs = ["convert_broadcast_in_dim.cc"],
    hdrs = ["convert_broadcast_in_dim.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_clamp",
    srcs = ["convert_clamp.cc"],
    hdrs = ["convert_clamp.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_concat",
    srcs = ["convert_concat.cc"],
    hdrs = ["convert_concat.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_cumsum",
    srcs = ["convert_cumsum.cc"],
    hdrs = ["convert_cumsum.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels/internal:tensor_ctypes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_fully_connected",
    srcs = ["convert_fully_connected.cc"],
    hdrs = ["convert_fully_connected.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:util",
    ],
)

cc_library(
    name = "convert_conv",
    srcs = ["convert_conv.cc"],
    hdrs = ["convert_conv.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_depth_to_space",
    srcs = ["convert_depth_to_space.cc"],
    hdrs = ["convert_depth_to_space.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_depthwise_conv",
    srcs = ["convert_depthwise_conv.cc"],
    hdrs = ["convert_depthwise_conv.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_dynamic_update_slice",
    srcs = ["convert_dynamic_update_slice.cc"],
    hdrs = ["convert_dynamic_update_slice.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_elementwise",
    srcs = ["convert_elementwise.cc"],
    hdrs = ["convert_elementwise.h"],
    deps = [
        ":convert_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "convert_embedding_lookup",
    srcs = ["convert_embedding_lookup.cc"],
    hdrs = ["convert_embedding_lookup.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_pad",
    srcs = ["convert_pad.cc"],
    hdrs = ["convert_pad.h"],
    deps = [
        ":convert_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
    ],
)

cc_library(
    name = "convert_transpose_conv",
    srcs = ["convert_transpose_conv.cc"],
    hdrs = ["convert_transpose_conv.h"],
    deps = [
        ":convert_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_unpack",
    srcs = ["convert_unpack.cc"],
    hdrs = ["convert_unpack.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_group_norm",
    srcs = ["convert_group_norm.cc"],
    hdrs = ["convert_group_norm.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_layer_norm",
    srcs = ["convert_layer_norm.cc"],
    hdrs = ["convert_layer_norm.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_one_hot",
    srcs = ["convert_one_hot.cc"],
    hdrs = ["convert_one_hot.h"],
    deps = [
        "//tflite/c:common",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_pack",
    srcs = ["convert_pack.cc"],
    hdrs = ["convert_pack.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_reverse",
    srcs = ["convert_reverse.cc"],
    hdrs = ["convert_reverse.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_pixel_shuffle",
    srcs = ["convert_pixel_shuffle.cc"],
    hdrs = ["convert_pixel_shuffle.h"],
    deps = [
        "//tflite/c:common",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_pooling2d",
    srcs = ["convert_pooling2d.cc"],
    hdrs = ["convert_pooling2d.h"],
    deps = [
        ":convert_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_relu",
    srcs = ["convert_relu.cc"],
    hdrs = ["convert_relu.h"],
    deps = [
        ":convert_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_reduce",
    srcs = ["convert_reduce.cc"],
    hdrs = ["convert_reduce.h"],
    deps = [
        ":convert_aux",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor_ctypes",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "convert_reshape",
    srcs = ["convert_reshape.cc"],
    hdrs = ["convert_reshape.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_resize2d",
    srcs = ["convert_resize2d.cc"],
    hdrs = ["convert_resize2d.h"],
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "convert_rms_norm",
    srcs = ["convert_rms_norm.cc"],
    hdrs = ["convert_rms_norm.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_slice",
    srcs = ["convert_slice.cc"],
    hdrs = ["convert_slice.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_softmax",
    srcs = ["convert_softmax.cc"],
    hdrs = ["convert_softmax.h"],
    deps = [
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
    ],
)

cc_library(
    name = "convert_select",
    srcs = ["convert_select.cc"],
    hdrs = ["convert_select.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_sdpa",
    srcs = ["convert_sdpa.cc"],
    hdrs = ["convert_sdpa.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers//include/flatbuffers:flexbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_space_to_depth",
    srcs = ["convert_space_to_depth.cc"],
    hdrs = ["convert_space_to_depth.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_split",
    srcs = ["convert_split.cc"],
    hdrs = ["convert_split.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_strided_slice",
    srcs = ["convert_strided_slice.cc"],
    hdrs = ["convert_strided_slice.h"],
    deps = [
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_tile",
    srcs = ["convert_tile.cc"],
    hdrs = ["convert_tile.h"],
    deps = [
        "//tflite/c:common",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_transpose",
    srcs = ["convert_transpose.cc"],
    hdrs = ["convert_transpose.h"],
    deps = [
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor_ctypes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_testing_utils",
    testonly = True,
    srcs = ["convert_testing_utils.cc"],
    hdrs = ["convert_testing_utils.h"],
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "stub_delegate",
    testonly = True,
    srcs = ["stub_delegate.cc"],
    hdrs = ["stub_delegate.h"],
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "//third_party/odml/litert/ml_drift/tflite/support",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_library(
    name = "convert_unpooling2d",
    srcs = ["convert_unpooling2d.cc"],
    hdrs = ["convert_unpooling2d.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_cbrt",
    srcs = ["convert_cbrt.cc"],
    hdrs = ["convert_cbrt.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_prelu",
    srcs = ["convert_prelu.cc"],
    hdrs = ["convert_prelu.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "convert_gather",
    srcs = ["convert_gather.cc"],
    hdrs = ["convert_gather.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_topk",
    srcs = ["convert_topk.cc"],
    hdrs = ["convert_topk.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_resampler",
    srcs = ["convert_resampler.cc"],
    hdrs = ["convert_resampler.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_rope",
    srcs = ["convert_rope.cc"],
    hdrs = ["convert_rope.h"],
    deps = [
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_argmax",
    srcs = ["convert_argmax.cc"],
    hdrs = ["convert_argmax.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_bitcast",
    srcs = ["convert_bitcast.cc"],
    hdrs = ["convert_bitcast.h"],
    deps = [
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_library(
    name = "convert_quantize",
    srcs = ["convert_quantize.cc"],
    hdrs = ["convert_quantize.h"],
    deps = [
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_library(
    name = "convert_dequantize",
    srcs = ["convert_dequantize.cc"],
    hdrs = ["convert_dequantize.h"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_absolute_positional_embedding_test",
    srcs = ["convert_absolute_positional_embedding_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "convert_aux_test",
    srcs = ["convert_aux_test.cc"],
    deps = [
        ":convert_aux",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
    ],
)

cc_test(
    name = "convert_batch_matmul_test",
    srcs = ["convert_batch_matmul_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_broadcast_in_dim_test",
    srcs = ["convert_broadcast_in_dim_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_clamp_test",
    srcs = ["convert_clamp_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "convert_concat_test",
    srcs = ["convert_concat_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_cumsum_test",
    srcs = ["convert_cumsum_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_fully_connected_test",
    srcs = ["convert_fully_connected_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_conv_test",
    srcs = ["convert_conv_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_depth_to_space_test",
    srcs = ["convert_depth_to_space_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:c_api_gms",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_depthwise_conv_test",
    srcs = ["convert_depthwise_conv_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_elementwise_test",
    srcs = ["convert_elementwise_test.cc"],
    deps = [
        ":convert_elementwise",
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_embedding_lookup_test",
    srcs = ["convert_embedding_lookup_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_pad_test",
    srcs = ["convert_pad_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
    ],
)

cc_test(
    name = "convert_transpose_conv_test",
    srcs = ["convert_transpose_conv_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_unpack_test",
    srcs = ["convert_unpack_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_group_norm_test",
    srcs = ["convert_group_norm_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_layer_norm_test",
    srcs = ["convert_layer_norm_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_one_hot_test",
    srcs = ["convert_one_hot_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_pack_test",
    srcs = ["convert_pack_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_reverse_test",
    srcs = ["convert_reverse_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_pixel_shuffle_test",
    srcs = ["convert_pixel_shuffle_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_pooling2d_test",
    srcs = ["convert_pooling2d_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_relu_test",
    srcs = ["convert_relu_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_reduce_test",
    srcs = ["convert_reduce_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_reshape_test",
    srcs = ["convert_reshape_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_resize2d_test",
    srcs = ["convert_resize2d_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_rms_norm_test",
    srcs = ["convert_rms_norm_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_slice_test",
    srcs = ["convert_slice_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_softmax_test",
    srcs = ["convert_softmax_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_space_to_depth_test",
    srcs = ["convert_space_to_depth_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:c_api_gms",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_split_test",
    srcs = ["convert_split_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_strided_slice_test",
    srcs = ["convert_strided_slice_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_tile_test",
    srcs = ["convert_tile_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "convert_transpose_test",
    srcs = ["convert_transpose_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_dynamic_update_slice_test",
    srcs = ["convert_dynamic_update_slice_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_select_test",
    srcs = ["convert_select_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "convert_sdpa_test",
    srcs = ["convert_sdpa_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//include/flatbuffers:flexbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_unpooling2d_test",
    srcs = ["convert_unpooling2d_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_cbrt_test",
    srcs = ["convert_cbrt_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_prelu_test",
    srcs = ["convert_prelu_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "convert_gather_test",
    srcs = ["convert_gather_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_topk_test",
    srcs = ["convert_topk_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_resampler_test",
    srcs = ["convert_resampler_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "convert_rope_test",
    srcs = ["convert_rope_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "convert_argmax_test",
    srcs = ["convert_argmax_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_bitcast_test",
    srcs = ["convert_bitcast_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
    ],
)

cc_test(
    name = "convert_quantize_test",
    srcs = ["convert_quantize_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)

cc_test(
    name = "convert_dequantize_test",
    srcs = ["convert_dequantize_test.cc"],
    deps = [
        ":convert_testing_utils",
        ":stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
    ],
)
