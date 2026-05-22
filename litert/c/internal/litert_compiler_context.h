// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif

/// A function table that contains LiteRT C APIs needed for Compiler Plugins.
///
/// @note This struct is shared with LiteRT runtime and Compiler Plugins. So it
/// must be ABI stable.
typedef struct LiteRtCompilerContext {
  // Model inspection
  LiteRtStatus (*get_num_model_subgraphs)(LiteRtModel model,
                                          LiteRtParamIndex* num_subgraphs);
  LiteRtStatus (*get_model_subgraph)(LiteRtModel model,
                                     LiteRtParamIndex subgraph_index,
                                     LiteRtSubgraph* subgraph);

  // Subgraph inspection
  LiteRtStatus (*get_num_subgraph_ops)(LiteRtSubgraph subgraph,
                                       LiteRtParamIndex* num_ops);
  LiteRtStatus (*get_subgraph_op)(LiteRtSubgraph subgraph,
                                  LiteRtParamIndex op_index, LiteRtOp* op);
  LiteRtStatus (*get_num_subgraph_inputs)(LiteRtSubgraph subgraph,
                                          LiteRtParamIndex* num_inputs);
  LiteRtStatus (*get_subgraph_input)(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex input_index,
                                     LiteRtTensor* input);
  LiteRtStatus (*get_num_subgraph_outputs)(LiteRtSubgraph subgraph,
                                           LiteRtParamIndex* num_outputs);
  LiteRtStatus (*get_subgraph_output)(LiteRtSubgraph subgraph,
                                      LiteRtParamIndex output_index,
                                      LiteRtTensor* output);

  // Op inspection
  LiteRtStatus (*get_op_code)(LiteRtOp op, LiteRtOpCode* code);
  LiteRtStatus (*get_custom_code)(LiteRtOp op, const char** code);
  LiteRtStatus (*get_num_op_inputs)(LiteRtOp op, LiteRtParamIndex* num_inputs);
  LiteRtStatus (*get_op_input)(LiteRtOp op, LiteRtParamIndex input_index,
                               LiteRtTensor* input);
  LiteRtStatus (*get_num_op_outputs)(LiteRtOp op,
                                     LiteRtParamIndex* num_outputs);
  LiteRtStatus (*get_op_output)(LiteRtOp op, LiteRtParamIndex output_index,
                                LiteRtTensor* output);

  // Tensor inspection
  LiteRtStatus (*get_tensor_name)(LiteRtTensor tensor, const char** name);
  LiteRtStatus (*get_tensor_index)(LiteRtTensor tensor, uint32_t* tensor_index);
  LiteRtStatus (*get_tensor_type_id)(LiteRtTensor tensor,
                                     LiteRtTensorTypeId* type_id);
  LiteRtStatus (*get_ranked_tensor_type)(
      LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type);
  LiteRtStatus (*get_unranked_tensor_type)(
      LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type);
  LiteRtStatus (*get_quantization_type_id)(LiteRtTensor tensor,
                                           LiteRtQuantizationTypeId* q_type_id);
  LiteRtStatus (*get_per_tensor_quantization)(
      LiteRtTensor tensor,
      LiteRtQuantizationPerTensor* per_tensor_quantization);
  LiteRtStatus (*get_per_channel_quantization)(
      LiteRtTensor tensor,
      LiteRtQuantizationPerChannel* per_channel_quantization);
  LiteRtStatus (*get_num_tensor_uses)(LiteRtTensor tensor,
                                      LiteRtParamIndex* num_uses);
  LiteRtStatus (*get_tensor_use)(LiteRtTensor tensor,
                                 LiteRtParamIndex use_index, LiteRtOp* user,
                                 LiteRtParamIndex* user_arg_index);
  LiteRtStatus (*get_tensor_defining_op)(LiteRtTensor tensor,
                                         bool* has_defining_op,
                                         LiteRtTensorDefiningOp* defining_op);

  // Weights
  LiteRtStatus (*get_tensor_weights)(LiteRtTensor tensor,
                                     LiteRtWeights* weights);
  LiteRtStatus (*get_weights_buffer_id)(LiteRtWeights weights,
                                        int32_t* buffer_id);
  LiteRtStatus (*get_weights_bytes)(LiteRtWeights weights, const void** addr,
                                    size_t* size);

  // Op options
  LiteRtStatus (*get_shlo_composite_op_name)(LiteRtOp op, const char** name);
  LiteRtStatus (*get_shlo_composite_op_decomposition_subgraph_index)(
      LiteRtOp op, int32_t* decomposition_subgraph_index);
  LiteRtStatus (*get_shlo_composite_op_attributes)(LiteRtOp op,
                                                   const uint8_t** attributes,
                                                   int32_t* attributes_size);
  LiteRtStatus (*get_shlo_composite_op_version)(LiteRtOp op, int32_t* version);

  // Utility
  LiteRtStatus (*push_op)(LiteRtOpList op_list, LiteRtOp op,
                          LiteRtParamIndex partition_index);

  // Options
  LiteRtStatus (*get_opaque_options)(LiteRtOptions options,
                                     LiteRtOpaqueOptions* opaque_options);
  LiteRtStatus (*find_opaque_options_data)(LiteRtOpaqueOptions options,
                                           const char* payload_identifier,
                                           void** payload_data);
  void (*destroy_options)(LiteRtOptions options);

  // Environment options
  LiteRtStatus (*get_environment_options_value)(
      LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
      LiteRtAny* value);

  // StridedSlice options
  LiteRtStatus (*get_strided_slice_begin_mask_option)(LiteRtOp op,
                                                      int32_t* begin_mask);
  LiteRtStatus (*get_strided_slice_end_mask_option)(LiteRtOp op,
                                                    int32_t* end_mask);
  LiteRtStatus (*get_strided_slice_shrink_axis_mask_option)(
      LiteRtOp op, int32_t* shrink_axis_mask);
  LiteRtStatus (*get_strided_slice_ellipsis_mask_option)(
      LiteRtOp op, int32_t* ellipsis_mask);
  LiteRtStatus (*get_strided_slice_new_axis_mask_option)(
      LiteRtOp op, int32_t* new_axis_mask);
  LiteRtStatus (*get_strided_slice_offset_option)(LiteRtOp op, bool* offset);

  // Sub options
  LiteRtStatus (*get_sub_fused_activation_option)(LiteRtOp op,
                                                  uint32_t* fused_activation);

  // Sum options
  LiteRtStatus (*get_sum_keep_dims_option)(LiteRtOp op, bool* keepdims);

  // Conv2d options
  LiteRtStatus (*get_conv_2d_padding_option)(LiteRtOp op, uint32_t* padding);
  LiteRtStatus (*get_conv_2d_stride_w_option)(LiteRtOp op, int32_t* stride_w);
  LiteRtStatus (*get_conv_2d_stride_h_option)(LiteRtOp op, int32_t* stride_h);
  LiteRtStatus (*get_conv_2d_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation_function);
  LiteRtStatus (*get_conv_2d_dilation_w_option)(LiteRtOp op,
                                                int32_t* dilation_w_factor);
  LiteRtStatus (*get_conv_2d_dilation_h_option)(LiteRtOp op,
                                                int32_t* dilation_h_factor);

  // DepthwiseConv2d options
  LiteRtStatus (*get_depthwise_conv_2d_padding_option)(LiteRtOp op,
                                                       uint32_t* padding);
  LiteRtStatus (*get_depthwise_conv_2d_stride_w_option)(LiteRtOp op,
                                                        int32_t* stride_w);
  LiteRtStatus (*get_depthwise_conv_2d_stride_h_option)(LiteRtOp op,
                                                        int32_t* stride_h);
  LiteRtStatus (*get_depthwise_conv_2d_depth_multiplier_option)(
      LiteRtOp op, int32_t* depth_multiplier);
  LiteRtStatus (*get_depthwise_conv_2d_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation_function);
  LiteRtStatus (*get_depthwise_conv_2d_dilation_w_option)(
      LiteRtOp op, int32_t* dilation_w_factor);
  LiteRtStatus (*get_depthwise_conv_2d_dilation_h_option)(
      LiteRtOp op, int32_t* dilation_h_factor);

  // TransposeConv options
  LiteRtStatus (*get_transpose_conv_padding_option)(LiteRtOp op,
                                                    uint32_t* padding);
  LiteRtStatus (*get_transpose_conv_stride_w_option)(LiteRtOp op,
                                                     int32_t* stride_w);
  LiteRtStatus (*get_transpose_conv_stride_h_option)(LiteRtOp op,
                                                     int32_t* stride_h);
  LiteRtStatus (*get_transpose_conv_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation_function);

  // Add options
  LiteRtStatus (*get_add_fused_activation_option)(LiteRtOp op,
                                                  uint32_t* fused_activation);

  // Mul options
  LiteRtStatus (*get_mul_fused_activation_option)(LiteRtOp op,
                                                  uint32_t* fused_activation);

  // Div options
  LiteRtStatus (*get_div_fused_activation_option)(LiteRtOp op,
                                                  uint32_t* fused_activation);

  // FullyConnected options
  LiteRtStatus (*get_fully_connected_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation);
  LiteRtStatus (*get_fully_connected_weights_format_option)(
      LiteRtOp op, uint32_t* weights_format);
  LiteRtStatus (*get_fully_connected_keep_num_dims_option)(LiteRtOp op,
                                                           bool* keep_num_dims);
  LiteRtStatus (*get_fully_connected_quantized_bias_type_option)(
      LiteRtOp op, uint32_t* quantized_bias_type);
  LiteRtStatus (*get_fully_connected_asymmetric_quantize_input_option)(
      LiteRtOp op, bool* asymmetric_quantize_input);

  // Softmax options
  LiteRtStatus (*get_softmax_beta_option)(LiteRtOp op, float* beta);

  // Concatenation options
  LiteRtStatus (*get_concatenation_axis_option)(LiteRtOp op, int32_t* axis);
  LiteRtStatus (*get_concatenation_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation);

  // Split options
  LiteRtStatus (*get_split_num_splits_option)(LiteRtOp op, int32_t* num_splits);

  // Mean options
  LiteRtStatus (*get_mean_keep_dims_option)(LiteRtOp op, bool* keepdims);

  // ReduceMax options
  LiteRtStatus (*get_reduce_max_keep_dims_option)(LiteRtOp op, bool* keepdims);

  // ResizeBilinear options
  LiteRtStatus (*get_resize_bilinear_align_corners_option)(LiteRtOp op,
                                                           bool* align_corners);
  LiteRtStatus (*get_resize_bilinear_half_pixel_center_option)(
      LiteRtOp op, bool* half_pixel_centers);

  // ResizeNearestNeighbor options
  LiteRtStatus (*get_resize_nearest_neighbor_align_corners_option)(
      LiteRtOp op, bool* align_corners);
  LiteRtStatus (*get_resize_nearest_neighbor_half_pixel_center_option)(
      LiteRtOp op, bool* half_pixel_centers);

  // BatchMatmul options
  LiteRtStatus (*get_batch_matmul_adj_x_option)(LiteRtOp op, bool* adj_x);
  LiteRtStatus (*get_batch_matmul_adj_y_option)(LiteRtOp op, bool* adj_y);
  LiteRtStatus (*get_batch_matmul_asymmetric_quantize_input_option)(
      LiteRtOp op, bool* asymmetric_quantize_input);

  // AveragePool2d options
  LiteRtStatus (*get_average_pool_2d_padding_option)(LiteRtOp op,
                                                     uint32_t* padding);
  LiteRtStatus (*get_average_pool_2d_stride_w_option)(LiteRtOp op,
                                                      int32_t* stride_w);
  LiteRtStatus (*get_average_pool_2d_stride_h_option)(LiteRtOp op,
                                                      int32_t* stride_h);
  LiteRtStatus (*get_average_pool_2d_filter_width_option)(
      LiteRtOp op, int32_t* filter_width);
  LiteRtStatus (*get_average_pool_2d_filter_height_option)(
      LiteRtOp op, int32_t* filter_height);
  LiteRtStatus (*get_average_pool_2d_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation);

  // MaxPool2d options
  LiteRtStatus (*get_max_pool_2d_padding_option)(LiteRtOp op,
                                                 uint32_t* padding);
  LiteRtStatus (*get_max_pool_2d_stride_w_option)(LiteRtOp op,
                                                  int32_t* stride_w);
  LiteRtStatus (*get_max_pool_2d_stride_h_option)(LiteRtOp op,
                                                  int32_t* stride_h);
  LiteRtStatus (*get_max_pool_2d_filter_width_option)(LiteRtOp op,
                                                      int32_t* filter_width);
  LiteRtStatus (*get_max_pool_2d_filter_height_option)(LiteRtOp op,
                                                       int32_t* filter_height);
  LiteRtStatus (*get_max_pool_2d_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation);

  // LeakyRelu options
  LiteRtStatus (*get_leaky_relu_alpha_option)(LiteRtOp op, float* alpha);

  // Reshape options
  LiteRtStatus (*get_reshape_new_shape_option)(LiteRtOp op,
                                               const int32_t** new_shape,
                                               int32_t* new_shape_size);

  // ReduceMin options
  LiteRtStatus (*get_reduce_min_keep_dims_option)(LiteRtOp op, bool* keepdims);

  // ReduceAny options
  LiteRtStatus (*get_reduce_any_keep_dims_option)(LiteRtOp op, bool* keepdims);

  // ReduceAll options
  LiteRtStatus (*get_reduce_all_keep_dims_option)(LiteRtOp op, bool* keepdims);

  // Pack options
  LiteRtStatus (*get_pack_axis_option)(LiteRtOp op, int32_t* axis);
  LiteRtStatus (*get_pack_values_count_option)(LiteRtOp op,
                                               int32_t* values_count);

  // OneHot options
  LiteRtStatus (*get_one_hot_axis_option)(LiteRtOp op, int32_t* axis);

  // Unpack options
  LiteRtStatus (*get_unpack_axis_option)(LiteRtOp op, int32_t* axis);
  LiteRtStatus (*get_unpack_num_option)(LiteRtOp op, int32_t* num);

  // Gather options
  LiteRtStatus (*get_gather_axis_option)(LiteRtOp op, int32_t* axis);
  LiteRtStatus (*get_gather_batch_dims_option)(LiteRtOp op,
                                               int32_t* batch_dims);

  // Conv3d options
  LiteRtStatus (*get_conv_3d_padding_option)(LiteRtOp op, uint32_t* padding);
  LiteRtStatus (*get_conv_3d_stride_d_option)(LiteRtOp op, int32_t* stride_d);
  LiteRtStatus (*get_conv_3d_stride_w_option)(LiteRtOp op, int32_t* stride_w);
  LiteRtStatus (*get_conv_3d_stride_h_option)(LiteRtOp op, int32_t* stride_h);
  LiteRtStatus (*get_conv_3d_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation_function);
  LiteRtStatus (*get_conv_3d_dilation_d_option)(LiteRtOp op,
                                                int32_t* dilation_d_factor);
  LiteRtStatus (*get_conv_3d_dilation_w_option)(LiteRtOp op,
                                                int32_t* dilation_w_factor);
  LiteRtStatus (*get_conv_3d_dilation_h_option)(LiteRtOp op,
                                                int32_t* dilation_h_factor);

  // L2Pool2d options
  LiteRtStatus (*get_l2_pool_2d_padding_option)(LiteRtOp op, uint32_t* padding);
  LiteRtStatus (*get_l2_pool_2d_stride_w_option)(LiteRtOp op,
                                                 int32_t* stride_w);
  LiteRtStatus (*get_l2_pool_2d_stride_h_option)(LiteRtOp op,
                                                 int32_t* stride_h);
  LiteRtStatus (*get_l2_pool_2d_filter_width_option)(LiteRtOp op,
                                                     int32_t* filter_width);
  LiteRtStatus (*get_l2_pool_2d_filter_height_option)(LiteRtOp op,
                                                      int32_t* filter_height);
  LiteRtStatus (*get_l2_pool_2d_fused_activation_option)(
      LiteRtOp op, uint32_t* fused_activation_function);

  // DepthToSpace options
  LiteRtStatus (*get_depth_to_space_block_size_option)(LiteRtOp op,
                                                       int32_t* block_size);

  // SpaceToDepth options
  LiteRtStatus (*get_space_to_depth_block_size_option)(LiteRtOp op,
                                                       int32_t* block_size);

  // CumSum options
  LiteRtStatus (*get_cumsum_exclusive_option)(LiteRtOp op, bool* exclusive);
  LiteRtStatus (*get_cumsum_reverse_option)(LiteRtOp op, bool* reverse);

  // Gelu options
  LiteRtStatus (*get_gelu_approximate_option)(LiteRtOp op, bool* approximate);

  // MirrorPad options
  LiteRtStatus (*get_mirror_pad_mode_option)(LiteRtOp op, uint32_t* mode);

  // Squeeze options
  LiteRtStatus (*get_squeeze_dims_option)(LiteRtOp op,
                                          const int32_t** squeeze_dims,
                                          int32_t* num_squeeze_dims);

} LiteRtCompilerContext;

// ABI compatibility check for LiteRtCompilerContext.
//
// Note: Please get review from the LiteRT ABI compatibility team when you make
// changes to this struct.
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(LiteRtCompilerContext) == 1024,
              "LiteRtCompilerContext size mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_model_subgraphs) == 0,
              "LiteRtCompilerContext get_num_model_subgraphs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_model_subgraph) == 8,
              "LiteRtCompilerContext get_model_subgraph offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_subgraph_ops) == 16,
              "LiteRtCompilerContext get_num_subgraph_ops offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_subgraph_op) == 24,
              "LiteRtCompilerContext get_subgraph_op offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_subgraph_inputs) == 32,
              "LiteRtCompilerContext get_num_subgraph_inputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_subgraph_input) == 40,
              "LiteRtCompilerContext get_subgraph_input offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_subgraph_outputs) == 48,
              "LiteRtCompilerContext get_num_subgraph_outputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_subgraph_output) == 56,
              "LiteRtCompilerContext get_subgraph_output offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_op_code) == 64,
              "LiteRtCompilerContext get_op_code offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_custom_code) == 72,
              "LiteRtCompilerContext get_custom_code offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_op_inputs) == 80,
              "LiteRtCompilerContext get_num_op_inputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_op_input) == 88,
              "LiteRtCompilerContext get_op_input offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_op_outputs) == 96,
              "LiteRtCompilerContext get_num_op_outputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_op_output) == 104,
              "LiteRtCompilerContext get_op_output offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_name) == 112,
              "LiteRtCompilerContext get_tensor_name offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_index) == 120,
              "LiteRtCompilerContext get_tensor_index offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_type_id) == 128,
              "LiteRtCompilerContext get_tensor_type_id offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_ranked_tensor_type) == 136,
              "LiteRtCompilerContext get_ranked_tensor_type offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_unranked_tensor_type) == 144,
              "LiteRtCompilerContext get_unranked_tensor_type offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_quantization_type_id) == 152,
              "LiteRtCompilerContext get_quantization_type_id offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_per_tensor_quantization) == 160,
    "LiteRtCompilerContext get_per_tensor_quantization offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_per_channel_quantization) == 168,
    "LiteRtCompilerContext get_per_channel_quantization offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_tensor_uses) == 176,
              "LiteRtCompilerContext get_num_tensor_uses offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_use) == 184,
              "LiteRtCompilerContext get_tensor_use offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_defining_op) == 192,
              "LiteRtCompilerContext get_tensor_defining_op offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_weights) == 200,
              "LiteRtCompilerContext get_tensor_weights offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_weights_buffer_id) == 208,
              "LiteRtCompilerContext get_weights_buffer_id offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_weights_bytes) == 216,
              "LiteRtCompilerContext get_weights_bytes offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_shlo_composite_op_name) == 224,
    "LiteRtCompilerContext get_shlo_composite_op_name offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_shlo_composite_op_decomposition_subgraph_index) == 232,
    "LiteRtCompilerContext get_shlo_composite_op_decomposition_subgraph_index "
    "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_shlo_composite_op_attributes) == 240,
    "LiteRtCompilerContext get_shlo_composite_op_attributes offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_shlo_composite_op_version) == 248,
    "LiteRtCompilerContext get_shlo_composite_op_version offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, push_op) == 256,
              "LiteRtCompilerContext push_op offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_opaque_options) == 264,
              "LiteRtCompilerContext get_opaque_options offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, find_opaque_options_data) == 272,
              "LiteRtCompilerContext find_opaque_options_data offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, destroy_options) == 280,
              "LiteRtCompilerContext destroy_options offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_environment_options_value) == 288,
    "LiteRtCompilerContext get_environment_options_value offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_strided_slice_begin_mask_option) == 296,
              "LiteRtCompilerContext get_strided_slice_begin_mask_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_strided_slice_end_mask_option) == 304,
    "LiteRtCompilerContext get_strided_slice_end_mask_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_strided_slice_shrink_axis_mask_option) == 312,
              "LiteRtCompilerContext get_strided_slice_shrink_axis_mask_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_strided_slice_ellipsis_mask_option) == 320,
              "LiteRtCompilerContext get_strided_slice_ellipsis_mask_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_strided_slice_new_axis_mask_option) == 328,
              "LiteRtCompilerContext get_strided_slice_new_axis_mask_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_strided_slice_offset_option) == 336,
    "LiteRtCompilerContext get_strided_slice_offset_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_sub_fused_activation_option) == 344,
    "LiteRtCompilerContext get_sub_fused_activation_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_sum_keep_dims_option) == 352,
              "LiteRtCompilerContext get_sum_keep_dims_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_2d_padding_option) == 360,
    "LiteRtCompilerContext get_conv_2d_padding_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_2d_stride_w_option) == 368,
    "LiteRtCompilerContext get_conv_2d_stride_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_2d_stride_h_option) == 376,
    "LiteRtCompilerContext get_conv_2d_stride_h_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_conv_2d_fused_activation_option) == 384,
              "LiteRtCompilerContext get_conv_2d_fused_activation_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_2d_dilation_w_option) == 392,
    "LiteRtCompilerContext get_conv_2d_dilation_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_2d_dilation_h_option) == 400,
    "LiteRtCompilerContext get_conv_2d_dilation_h_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_padding_option) == 408,
              "LiteRtCompilerContext get_depthwise_conv_2d_padding_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_stride_w_option) == 416,
              "LiteRtCompilerContext get_depthwise_conv_2d_stride_w_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_stride_h_option) == 424,
              "LiteRtCompilerContext get_depthwise_conv_2d_stride_h_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_depth_multiplier_option) == 432,
              "LiteRtCompilerContext "
              "get_depthwise_conv_2d_depth_multiplier_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_fused_activation_option) == 440,
              "LiteRtCompilerContext "
              "get_depthwise_conv_2d_fused_activation_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_dilation_w_option) == 448,
              "LiteRtCompilerContext get_depthwise_conv_2d_dilation_w_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_depthwise_conv_2d_dilation_h_option) == 456,
              "LiteRtCompilerContext get_depthwise_conv_2d_dilation_h_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_transpose_conv_padding_option) == 464,
    "LiteRtCompilerContext get_transpose_conv_padding_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_transpose_conv_stride_w_option) == 472,
    "LiteRtCompilerContext get_transpose_conv_stride_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_transpose_conv_stride_h_option) == 480,
    "LiteRtCompilerContext get_transpose_conv_stride_h_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_transpose_conv_fused_activation_option) == 488,
              "LiteRtCompilerContext "
              "get_transpose_conv_fused_activation_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_add_fused_activation_option) == 496,
    "LiteRtCompilerContext get_add_fused_activation_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_mul_fused_activation_option) == 504,
    "LiteRtCompilerContext get_mul_fused_activation_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_div_fused_activation_option) == 512,
    "LiteRtCompilerContext get_div_fused_activation_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_fully_connected_fused_activation_option) == 520,
              "LiteRtCompilerContext "
              "get_fully_connected_fused_activation_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_fully_connected_weights_format_option) == 528,
              "LiteRtCompilerContext get_fully_connected_weights_format_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_fully_connected_keep_num_dims_option) == 536,
              "LiteRtCompilerContext get_fully_connected_keep_num_dims_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_fully_connected_quantized_bias_type_option) == 544,
              "LiteRtCompilerContext "
              "get_fully_connected_quantized_bias_type_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_fully_connected_asymmetric_quantize_input_option) == 552,
    "LiteRtCompilerContext "
    "get_fully_connected_asymmetric_quantize_input_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_softmax_beta_option) == 560,
              "LiteRtCompilerContext get_softmax_beta_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_concatenation_axis_option) == 568,
    "LiteRtCompilerContext get_concatenation_axis_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_concatenation_fused_activation_option) == 576,
              "LiteRtCompilerContext get_concatenation_fused_activation_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_split_num_splits_option) == 584,
    "LiteRtCompilerContext get_split_num_splits_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_mean_keep_dims_option) == 592,
    "LiteRtCompilerContext get_mean_keep_dims_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_reduce_max_keep_dims_option) == 600,
    "LiteRtCompilerContext get_reduce_max_keep_dims_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_resize_bilinear_align_corners_option) == 608,
              "LiteRtCompilerContext get_resize_bilinear_align_corners_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_resize_bilinear_half_pixel_center_option) == 616,
              "LiteRtCompilerContext "
              "get_resize_bilinear_half_pixel_center_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_resize_nearest_neighbor_align_corners_option) == 624,
    "LiteRtCompilerContext get_resize_nearest_neighbor_align_corners_option "
    "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_resize_nearest_neighbor_half_pixel_center_option) == 632,
    "LiteRtCompilerContext "
    "get_resize_nearest_neighbor_half_pixel_center_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_batch_matmul_adj_x_option) == 640,
    "LiteRtCompilerContext get_batch_matmul_adj_x_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_batch_matmul_adj_y_option) == 648,
    "LiteRtCompilerContext get_batch_matmul_adj_y_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_batch_matmul_asymmetric_quantize_input_option) == 656,
    "LiteRtCompilerContext get_batch_matmul_asymmetric_quantize_input_option "
    "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_average_pool_2d_padding_option) == 664,
    "LiteRtCompilerContext get_average_pool_2d_padding_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_stride_w_option) == 672,
              "LiteRtCompilerContext get_average_pool_2d_stride_w_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_stride_h_option) == 680,
              "LiteRtCompilerContext get_average_pool_2d_stride_h_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_filter_width_option) == 688,
              "LiteRtCompilerContext get_average_pool_2d_filter_width_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_filter_height_option) == 696,
              "LiteRtCompilerContext get_average_pool_2d_filter_height_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_fused_activation_option) == 704,
              "LiteRtCompilerContext "
              "get_average_pool_2d_fused_activation_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_max_pool_2d_padding_option) == 712,
    "LiteRtCompilerContext get_max_pool_2d_padding_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_max_pool_2d_stride_w_option) == 720,
    "LiteRtCompilerContext get_max_pool_2d_stride_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_max_pool_2d_stride_h_option) == 728,
    "LiteRtCompilerContext get_max_pool_2d_stride_h_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_max_pool_2d_filter_width_option) == 736,
              "LiteRtCompilerContext get_max_pool_2d_filter_width_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_max_pool_2d_filter_height_option) == 744,
              "LiteRtCompilerContext get_max_pool_2d_filter_height_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_max_pool_2d_fused_activation_option) == 752,
              "LiteRtCompilerContext get_max_pool_2d_fused_activation_option "
              "offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_leaky_relu_alpha_option) == 760,
    "LiteRtCompilerContext get_leaky_relu_alpha_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext,
                       get_fully_connected_keep_num_dims_option) == 536,
              "LiteRtCompilerContext get_fully_connected_keep_num_dims_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_fully_connected_quantized_bias_type_option) == 544,
              "LiteRtCompilerContext "
              "get_fully_connected_quantized_bias_type_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_fully_connected_asymmetric_quantize_input_option) == 552,
    "LiteRtCompilerContext "
    "get_fully_connected_asymmetric_quantize_input_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext, get_softmax_beta_option) == 560,
              "LiteRtCompilerContext get_softmax_beta_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_concatenation_axis_option) == 568,
    "LiteRtCompilerContext get_concatenation_axis_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_concatenation_fused_activation_option) == 576,
              "LiteRtCompilerContext get_concatenation_fused_activation_option "
              "offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_split_num_splits_option) == 584,
    "LiteRtCompilerContext get_split_num_splits_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_mean_keep_dims_option) == 592,
    "LiteRtCompilerContext get_mean_keep_dims_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_reduce_max_keep_dims_option) == 600,
    "LiteRtCompilerContext get_reduce_max_keep_dims_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext,
                       get_resize_bilinear_align_corners_option) == 608,
              "LiteRtCompilerContext get_resize_bilinear_align_corners_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_resize_bilinear_half_pixel_center_option) == 616,
              "LiteRtCompilerContext "
              "get_resize_bilinear_half_pixel_center_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext,
             get_resize_nearest_neighbor_align_corners_option) == 624,
    "LiteRtCompilerContext get_resize_nearest_neighbor_align_corners_option "
    "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_resize_nearest_neighbor_half_pixel_center_option) == 632,
    "LiteRtCompilerContext "
    "get_resize_nearest_neighbor_half_pixel_center_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_batch_matmul_adj_x_option) == 640,
    "LiteRtCompilerContext get_batch_matmul_adj_x_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_batch_matmul_adj_y_option) == 648,
    "LiteRtCompilerContext get_batch_matmul_adj_y_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_batch_matmul_asymmetric_quantize_input_option) == 656,
    "LiteRtCompilerContext get_batch_matmul_asymmetric_quantize_input_option "
    "offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_average_pool_2d_padding_option) == 664,
    "LiteRtCompilerContext get_average_pool_2d_padding_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_stride_w_option) == 672,
              "LiteRtCompilerContext get_average_pool_2d_stride_w_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_stride_h_option) == 680,
              "LiteRtCompilerContext get_average_pool_2d_stride_h_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_filter_width_option) == 688,
              "LiteRtCompilerContext get_average_pool_2d_filter_width_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_filter_height_option) == 696,
              "LiteRtCompilerContext get_average_pool_2d_filter_height_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_average_pool_2d_fused_activation_option) == 704,
              "LiteRtCompilerContext "
              "get_average_pool_2d_fused_activation_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_max_pool_2d_padding_option) == 712,
    "LiteRtCompilerContext get_max_pool_2d_padding_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_max_pool_2d_stride_w_option) == 720,
    "LiteRtCompilerContext get_max_pool_2d_stride_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_max_pool_2d_stride_h_option) == 728,
    "LiteRtCompilerContext get_max_pool_2d_stride_h_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_max_pool_2d_filter_width_option) == 736,
              "LiteRtCompilerContext get_max_pool_2d_filter_width_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_max_pool_2d_filter_height_option) == 744,
              "LiteRtCompilerContext get_max_pool_2d_filter_height_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_max_pool_2d_fused_activation_option) == 752,
              "LiteRtCompilerContext get_max_pool_2d_fused_activation_option "
              "offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_leaky_relu_alpha_option) == 760,
    "LiteRtCompilerContext get_leaky_relu_alpha_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_reshape_new_shape_option) == 768,
    "LiteRtCompilerContext get_reshape_new_shape_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_reduce_min_keep_dims_option) == 776,
    "LiteRtCompilerContext get_reduce_min_keep_dims_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_reduce_any_keep_dims_option) == 784,
    "LiteRtCompilerContext get_reduce_any_keep_dims_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_reduce_all_keep_dims_option) == 792,
    "LiteRtCompilerContext get_reduce_all_keep_dims_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext, get_pack_axis_option) == 800,
              "LiteRtCompilerContext get_pack_axis_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_pack_values_count_option) == 808,
    "LiteRtCompilerContext get_pack_values_count_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext, get_one_hot_axis_option) == 816,
              "LiteRtCompilerContext get_one_hot_axis_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext, get_unpack_axis_option) == 824,
              "LiteRtCompilerContext get_unpack_axis_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_unpack_num_option) == 832,
              "LiteRtCompilerContext get_unpack_num_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext, get_gather_axis_option) == 840,
              "LiteRtCompilerContext get_gather_axis_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_gather_batch_dims_option) == 848,
    "LiteRtCompilerContext get_gather_batch_dims_option offset mismatch");

static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_padding_option) == 856,
    "LiteRtCompilerContext get_conv_3d_padding_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_stride_d_option) == 864,
    "LiteRtCompilerContext get_conv_3d_stride_d_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_stride_w_option) == 872,
    "LiteRtCompilerContext get_conv_3d_stride_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_stride_h_option) == 880,
    "LiteRtCompilerContext get_conv_3d_stride_h_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_conv_3d_fused_activation_option) == 888,
              "LiteRtCompilerContext get_conv_3d_fused_activation_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_dilation_d_option) == 896,
    "LiteRtCompilerContext get_conv_3d_dilation_d_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_dilation_w_option) == 904,
    "LiteRtCompilerContext get_conv_3d_dilation_w_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_conv_3d_dilation_h_option) == 912,
    "LiteRtCompilerContext get_conv_3d_dilation_h_option offset mismatch");

static_assert(offsetof(LiteRtCompilerContext,
                       get_depth_to_space_block_size_option) == 968,
              "LiteRtCompilerContext get_depth_to_space_block_size_option "
              "offset mismatch");
static_assert(offsetof(LiteRtCompilerContext,
                       get_space_to_depth_block_size_option) == 976,
              "LiteRtCompilerContext get_space_to_depth_block_size_option "
              "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_cumsum_exclusive_option) == 984,
    "LiteRtCompilerContext get_cumsum_exclusive_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_cumsum_reverse_option) == 992,
    "LiteRtCompilerContext get_cumsum_reverse_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_gelu_approximate_option) == 1000,
    "LiteRtCompilerContext get_gelu_approximate_option offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_mirror_pad_mode_option) == 1008,
    "LiteRtCompilerContext get_mirror_pad_mode_option offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_squeeze_dims_option) == 1016,
              "LiteRtCompilerContext get_squeeze_dims_option offset mismatch");
#endif  // __cplusplus

LiteRtCompilerContext* LrtGetCompilerContext();

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_
