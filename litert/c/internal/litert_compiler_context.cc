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

#include "litert/c/internal/litert_compiler_context.h"

#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"

LiteRtCompilerContext* LrtGetCompilerContext() {
  static LiteRtCompilerContext ctx = {
      .get_num_model_subgraphs = LiteRtGetNumModelSubgraphs,
      .get_model_subgraph = LiteRtGetModelSubgraph,
      .get_num_subgraph_ops = LiteRtGetNumSubgraphOps,
      .get_subgraph_op = LiteRtGetSubgraphOp,
      .get_num_subgraph_inputs = LiteRtGetNumSubgraphInputs,
      .get_subgraph_input = LiteRtGetSubgraphInput,
      .get_num_subgraph_outputs = LiteRtGetNumSubgraphOutputs,
      .get_subgraph_output = LiteRtGetSubgraphOutput,
      .get_op_code = LiteRtGetOpCode,
      .get_custom_code = LiteRtGetCustomCode,
      .get_num_op_inputs = LiteRtGetNumOpInputs,
      .get_op_input = LiteRtGetOpInput,
      .get_num_op_outputs = LiteRtGetNumOpOutputs,
      .get_op_output = LiteRtGetOpOutput,
      .get_tensor_name = LiteRtGetTensorName,
      .get_tensor_index = LiteRtGetTensorIndex,
      .get_tensor_type_id = LiteRtGetTensorTypeId,
      .get_ranked_tensor_type = LiteRtGetRankedTensorType,
      .get_unranked_tensor_type = LiteRtGetUnrankedTensorType,
      .get_quantization_type_id = LiteRtGetQuantizationTypeId,
      .get_per_tensor_quantization = LiteRtGetPerTensorQuantization,
      .get_per_channel_quantization = LiteRtGetPerChannelQuantization,
      .get_num_tensor_uses = LiteRtGetNumTensorUses,
      .get_tensor_use = LiteRtGetTensorUse,
      .get_tensor_defining_op = LiteRtGetTensorDefiningOp,
      .get_tensor_weights = LiteRtGetTensorWeights,
      .get_weights_buffer_id = LiteRtGetWeightsBufferId,
      .get_weights_bytes = LiteRtGetWeightsBytes,
      .get_shlo_composite_op_name = LiteRtGetSHLOCompositeOpName,
      .get_shlo_composite_op_decomposition_subgraph_index =
          LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex,
      .get_shlo_composite_op_attributes = LiteRtGetSHLOCompositeOpAttributes,
      .get_shlo_composite_op_version = LiteRtGetSHLOCompositeOpVersion,
      .push_op = LiteRtPushOp,
      .get_opaque_options = LiteRtGetOpaqueOptions,
      .find_opaque_options_data = LiteRtFindOpaqueOptionsData,
      .destroy_options = LiteRtDestroyOptions,
      .get_environment_options_value = LiteRtGetEnvironmentOptionsValue,

      .get_strided_slice_begin_mask_option =
          LiteRtGetStridedSliceBeginMaskOption,
      .get_strided_slice_end_mask_option = LiteRtGetStridedSliceEndMaskOption,
      .get_strided_slice_shrink_axis_mask_option =
          LiteRtGetStridedSliceShrinkAxisMaskOption,
      .get_strided_slice_ellipsis_mask_option =
          LiteRtGetStridedSliceEllipsisMaskOption,
      .get_strided_slice_new_axis_mask_option =
          LiteRtGetStridedSliceNewAxisMaskOption,
      .get_strided_slice_offset_option = LiteRtGetStridedSliceOffsetOption,

      .get_sub_fused_activation_option = LiteRtGetSubFusedActivationOption,

      .get_sum_keep_dims_option = LiteRtGetSumKeepDimsOption,

      .get_conv_2d_padding_option = LiteRtGetConv2dPaddingOption,
      .get_conv_2d_stride_w_option = LiteRtGetConv2dStrideWOption,
      .get_conv_2d_stride_h_option = LiteRtGetConv2dStrideHOption,
      .get_conv_2d_fused_activation_option =
          LiteRtGetConv2dFusedActivationOption,
      .get_conv_2d_dilation_w_option = LiteRtGetConv2dDilationWOption,
      .get_conv_2d_dilation_h_option = LiteRtGetConv2dDilationHOption,

      .get_depthwise_conv_2d_padding_option =
          LiteRtGetDepthwiseConv2dPaddingOption,
      .get_depthwise_conv_2d_stride_w_option =
          LiteRtGetDepthwiseConv2dStrideWOption,
      .get_depthwise_conv_2d_stride_h_option =
          LiteRtGetDepthwiseConv2dStrideHOption,
      .get_depthwise_conv_2d_depth_multiplier_option =
          LiteRtGetDepthwiseConv2dDepthMultiplierOption,
      .get_depthwise_conv_2d_fused_activation_option =
          LiteRtGetDepthwiseConv2dFusedActivationOption,
      .get_depthwise_conv_2d_dilation_w_option =
          LiteRtGetDepthwiseConv2dDilationWOption,
      .get_depthwise_conv_2d_dilation_h_option =
          LiteRtGetDepthwiseConv2dDilationHOption,

      .get_transpose_conv_padding_option = LiteRtGetTransposeConvPaddingOption,
      .get_transpose_conv_stride_w_option = LiteRtGetTransposeConvStrideWOption,
      .get_transpose_conv_stride_h_option = LiteRtGetTransposeConvStrideHOption,
      .get_transpose_conv_fused_activation_option =
          LiteRtGetTransposeConvFusedActivationOption,

      .get_add_fused_activation_option = LiteRtGetAddFusedActivationOption,

      .get_mul_fused_activation_option = LiteRtGetMulFusedActivationOption,

      .get_div_fused_activation_option = LiteRtGetDivFusedActivationOption,

      .get_fully_connected_fused_activation_option =
          LiteRtGetFullyConnectedFusedActivationOption,
      .get_fully_connected_weights_format_option =
          LiteRtGetFullyConnectedWeightsFormatOption,
      .get_fully_connected_keep_num_dims_option =
          LiteRtGetFullyConnectedKeepNumDimsOption,
      .get_fully_connected_quantized_bias_type_option =
          LiteRtFullyConnectedGetQuantizedBiasTypeOption,
      .get_fully_connected_asymmetric_quantize_input_option =
          LiteRtGetFullyConnectedAsymmetricQuantizeInputOption,

      .get_softmax_beta_option = LiteRtGetSoftmaxBetaOption,

      .get_concatenation_axis_option = LiteRtGetConcatenationAxisOption,
      .get_concatenation_fused_activation_option =
          LiteRtGetConcatenationFusedActivationOption,

      .get_split_num_splits_option = LiteRtGetSplitNumSplitsOption,

      .get_mean_keep_dims_option = LiteRtGetMeanKeepDimsOption,

      .get_reduce_max_keep_dims_option = LiteRtGetReduceMaxKeepDimsOption,

      .get_resize_bilinear_align_corners_option =
          LiteRtGetResizeBilinearAlignCornersOption,
      .get_resize_bilinear_half_pixel_center_option =
          LiteRtGetResizeBilinearHalfPixelCenterOption,

      .get_resize_nearest_neighbor_align_corners_option =
          LiteRtGetResizeNearestNeighborAlignCornersOption,
      .get_resize_nearest_neighbor_half_pixel_center_option =
          LiteRtGetResizeNearestNeighborHalfPixelCenterOption,

      .get_batch_matmul_adj_x_option = LiteRtGetBatchMatmulAdjXOption,
      .get_batch_matmul_adj_y_option = LiteRtGetBatchMatmulAdjYOption,
      .get_batch_matmul_asymmetric_quantize_input_option =
          LiteRtGetBatchMatmulAsymmetricQuantizeInputOption,

      .get_average_pool_2d_padding_option = LiteRtGetAveragePool2dPaddingOption,
      .get_average_pool_2d_stride_w_option =
          LiteRtGetAveragePool2dStrideWOption,
      .get_average_pool_2d_stride_h_option =
          LiteRtGetAveragePool2dStrideHOption,
      .get_average_pool_2d_filter_width_option =
          LiteRtGetAveragePool2dFilterWidthOption,
      .get_average_pool_2d_filter_height_option =
          LiteRtGetAveragePool2dFilterHeightOption,
      .get_average_pool_2d_fused_activation_option =
          LiteRtGetAveragePool2dFusedActivationOption,

      .get_max_pool_2d_padding_option = LiteRtGetMaxPool2dPaddingOption,
      .get_max_pool_2d_stride_w_option = LiteRtGetMaxPool2dStrideWOption,
      .get_max_pool_2d_stride_h_option = LiteRtGetMaxPool2dStrideHOption,
      .get_max_pool_2d_filter_width_option =
          LiteRtGetMaxPool2dFilterWidthOption,
      .get_max_pool_2d_filter_height_option =
          LiteRtGetMaxPool2dFilterHeightOption,
      .get_max_pool_2d_fused_activation_option =
          LiteRtGetMaxPool2dFusedActivationOption,

      .get_leaky_relu_alpha_option = LiteRtGetLeakyReluAlphaOption,

      .get_reshape_new_shape_option = LiteRtGetReshapeNewShapeOption,

      .get_reduce_min_keep_dims_option = LiteRtGetReduceMinKeepDimsOption,

      .get_reduce_any_keep_dims_option = LiteRtGetReduceAnyKeepDimsOption,

      .get_reduce_all_keep_dims_option = LiteRtGetReduceAllKeepDimsOption,

      .get_pack_axis_option = LiteRtGetPackAxisOption,
      .get_pack_values_count_option = LiteRtGetPackValuesCountOption,

      .get_one_hot_axis_option = LiteRtGetOneHotAxisOption,

      .get_unpack_axis_option = LiteRtGetUnpackAxisOption,
      .get_unpack_num_option = LiteRtGetUnpackNumOption,

      .get_gather_axis_option = LiteRtGetGatherAxisOption,
      .get_gather_batch_dims_option = LiteRtGetGatherBatchDimsOption,

      .get_conv_3d_padding_option = LiteRtGetConv3dPaddingOption,
      .get_conv_3d_stride_d_option = LiteRtGetConv3dStrideDOption,
      .get_conv_3d_stride_w_option = LiteRtGetConv3dStrideWOption,
      .get_conv_3d_stride_h_option = LiteRtGetConv3dStrideHOption,
      .get_conv_3d_fused_activation_option =
          LiteRtGetConv3dFusedActivationOption,
      .get_conv_3d_dilation_d_option = LiteRtGetConv3dDilationDOption,
      .get_conv_3d_dilation_w_option = LiteRtGetConv3dDilationWOption,
      .get_conv_3d_dilation_h_option = LiteRtGetConv3dDilationHOption,

      .get_l2_pool_2d_padding_option = LiteRtGetL2Pool2dPaddingOption,
      .get_l2_pool_2d_stride_w_option = LiteRtGetL2Pool2dStrideWOption,
      .get_l2_pool_2d_stride_h_option = LiteRtGetL2Pool2dStrideHOption,
      .get_l2_pool_2d_filter_width_option = LiteRtGetL2Pool2dFilterWidthOption,
      .get_l2_pool_2d_filter_height_option =
          LiteRtGetL2Pool2dFilterHeightOption,
      .get_l2_pool_2d_fused_activation_option =
          LiteRtGetL2Pool2dFusedActivationOption,

      .get_depth_to_space_block_size_option =
          LiteRtGetDepthToSpaceBlockSizeOption,

      .get_space_to_depth_block_size_option =
          LiteRtGetSpaceToDepthBlockSizeOption,

      .get_cumsum_exclusive_option = LiteRtGetCumsumExclusiveOption,
      .get_cumsum_reverse_option = LiteRtGetCumsumReverseOption,

      .get_gelu_approximate_option = LiteRtGetGeluApproximateOption,

      .get_mirror_pad_mode_option = LiteRtGetMirrorPadModeOption,

      .get_squeeze_dims_option = LiteRtGetSqueezeDimsOption,

      .get_custom_options = LiteRtGetCustomOptions,
  };
  return &ctx;
}
