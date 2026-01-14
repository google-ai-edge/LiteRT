// Copyright 2024 Google LLC.
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

#include "litert/cc/internal/litert_op_options.h"

#include <cstdint>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_builder.h"  // IWYU pragma: keep
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

LiteRtStatus CompositeOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* op_name;
  LITERT_RETURN_IF_ERROR(LiteRtGetSHLOCompositeOpName(op, &op_name));
  name = op_name;

  LITERT_RETURN_IF_ERROR(
      LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex(op, &subgraph));

  LITERT_RETURN_IF_ERROR(LiteRtGetSHLOCompositeOpVersion(op, &version));

  const uint8_t* impl_attributes = nullptr;
  int32_t impl_attributes_size = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetSHLOCompositeOpAttributes(
      op, &impl_attributes, &impl_attributes_size));

  if (impl_attributes_size > 0) {
    attributes_map =
        flexbuffers::GetRoot(impl_attributes, impl_attributes_size).AsMap();
  }
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> CompositeOptions::SetOpOptions(LiteRtBuilder builder) {
  return Unexpected(kLiteRtStatusErrorUnsupported);
}

LiteRtStatus RmsNormOpts::InitFromOp(LiteRtOp litert_op) {
  LITERT_RETURN_IF_ERROR(CompositeOptions::InitFromOp(litert_op));
  if (!attributes_map.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  constexpr char kEpsilonKey[] = "epsilon";
  flexbuffers::Reference raw_epsilon = attributes_map.value()[kEpsilonKey];
  if (raw_epsilon.IsNull()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  epsilon = raw_epsilon.AsFloat();
  return kLiteRtStatusOk;
}

Expected<void> RmsNormOpts::SetOpOptions(LiteRtBuilder builder) {
  return Unexpected(kLiteRtStatusErrorUnsupported);
}

LiteRtStatus AddOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAddFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> AddOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildAddOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus BatchMatmulOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjXOption(op, &adj_x));
  LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjYOption(op, &adj_y));
  LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> BatchMatmulOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildBatchMatmulOpOption(
      builder, op, &adj_x, &adj_y, &asymmetric_quantize_input));
  return Expected<void>();
}

LiteRtStatus ConcatenationOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationAxisOption(op, &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationFusedActivationOption(
      op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ConcatenationOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildConcatenationOpOption(
      builder, op, &fused_activation_function, &axis));
  return Expected<void>();
}

LiteRtStatus DivOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDivFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> DivOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildDivOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus FullyConnectedOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedFusedActivationOption(
      op, &fused_activation_function));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetFullyConnectedWeightsFormatOption(op, &weights_format));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetFullyConnectedKeepNumDimsOption(op, &keep_num_dims));
  uint32_t retrieved_quantized_bias_type;
  LITERT_RETURN_IF_ERROR(LiteRtFullyConnectedGetQuantizedBiasTypeOption(
      op, &retrieved_quantized_bias_type));
  quantized_bias_type = GetElementType(retrieved_quantized_bias_type);
  LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> FullyConnectedOptions::SetOpOptions(LiteRtBuilder builder) {
  uint32_t quantized_bias_type_uint32 =
      GetTfliteTensorType(quantized_bias_type);
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildFullyConnectedOpOption(
      builder, op, &fused_activation_function, &weights_format, &keep_num_dims,
      &quantized_bias_type_uint32, &asymmetric_quantize_input));
  return Expected<void>();
}

LiteRtStatus MulOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMulFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> MulOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildMulOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus SoftmaxOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSoftmaxBetaOption(op, &beta));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> SoftmaxOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildSoftmaxOpOption(builder, op, &beta));
  return Expected<void>();
}

LiteRtStatus StridedSliceOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceBeginMaskOption(op, &begin_mask));
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceEndMaskOption(op, &end_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceEllipsisMaskOption(op, &ellipsis_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceNewAxisMaskOption(op, &new_axis_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceShrinkAxisMaskOption(op, &shrink_axis_mask));
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceOffsetOption(op, &offset));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> StridedSliceOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildStridedSliceOpOption(
      builder, op, &begin_mask, &end_mask, &ellipsis_mask, &new_axis_mask,
      &shrink_axis_mask, &offset));
  return Expected<void>();
}

LiteRtStatus SubOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetSubFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> SubOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSubOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus ReshapeOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const int32_t* new_shape_data;
  int32_t new_shape_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetReshapeNewShapeOption(op, &new_shape_data, &new_shape_size));
  new_shape.assign(new_shape_data, new_shape_data + new_shape_size);

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ReshapeOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildReshapeOpOption(
      builder, op, new_shape.data(), new_shape.size()));
  return Expected<void>();
}

LiteRtStatus SumOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSumKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> SumOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSumOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

LiteRtStatus ReduceMaxOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceMax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceMaxKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ReduceMaxOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceMaxOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

LiteRtStatus ReduceMinOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceMin) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceMinKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ReduceMinOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceMinOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

LiteRtStatus ReduceAnyOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceAny) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceAnyKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ReduceAnyOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceAnyOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

LiteRtStatus ReduceAllOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceAll) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceAllKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ReduceAllOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceAllOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

LiteRtStatus PackOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflPack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(op, &axis));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> PackOptions::SetOpOptions(LiteRtBuilder builder) {
  LiteRtParamIndex num_inputs;
  LITERT_RETURN_IF_ERROR(LiteRtGetNumOpInputs(op, &num_inputs));
  int32_t values_count_int = static_cast<int32_t>(num_inputs);
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildPackOpOption(builder, op, &axis, &values_count_int));
  return Expected<void>();
}

LiteRtStatus UnpackOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflUnpack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetUnpackAxisOption(op, &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetUnpackNumOption(op, &num));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> UnpackOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildUnpackOpOption(builder, op, &axis, &num));
  return Expected<void>();
}

LiteRtStatus GatherOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(op, &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetGatherBatchDimsOption(op, &batch_dims));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> GatherOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildGatherOpOption(builder, op, &axis, &batch_dims));
  return Expected<void>();
}

LiteRtStatus MeanOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetMeanKeepDimsOption(op, &keep_dims));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> MeanOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildMeanOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

LiteRtStatus SplitOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSplit) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSplitNumSplitsOption(op, &num_splits));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> SplitOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSplitOpOption(builder, op, &num_splits));
  return Expected<void>();
}

LiteRtStatus Conv2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dDilationWOption(op, &dilation_w_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dDilationHOption(op, &dilation_h_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> Conv2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildConv2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &dilation_w_factor,
      &dilation_h_factor, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus Conv3dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dStrideDOption(op, &stride_d));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dDilationWOption(op, &dilation_w_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dDilationHOption(op, &dilation_h_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dDilationDOption(op, &dilation_d_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> Conv3dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildConv3dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &stride_d,
      &dilation_w_factor, &dilation_h_factor, &dilation_d_factor,
      &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus DepthwiseConv2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDepthMultiplierOption(op, &depth_multiplier));
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dFusedActivationOption(
      op, &fused_activation_function));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationWOption(op, &dilation_w_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationHOption(op, &dilation_h_factor));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> DepthwiseConv2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildDepthwiseConv2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &depth_multiplier,
      &fused_activation_function, &dilation_w_factor, &dilation_h_factor));
  return Expected<void>();
}

LiteRtStatus TransposeConvOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvFusedActivationOption(
      op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> TransposeConvOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildTransposeConvOpOption(
      builder, op, &padding, &stride_w, &stride_h, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus AveragePool2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterWidthOption(op, &filter_width));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterHeightOption(op, &filter_height));
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFusedActivationOption(
      op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> AveragePool2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildAveragePool2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus MaxPool2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterWidthOption(op, &filter_width));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterHeightOption(op, &filter_height));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> MaxPool2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildMaxPool2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus L2Pool2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflL2Pool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dFilterWidthOption(op, &filter_width));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetL2Pool2dFilterHeightOption(op, &filter_height));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetL2Pool2dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> L2Pool2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildL2Pool2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  return Expected<void>();
}

LiteRtStatus ResizeBilinearOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeBilinearAlignCornersOption(op, &align_corners));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeBilinearHalfPixelCenterOption(op, &half_pixel_centers));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ResizeBilinearOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildResizeBilinearOpOption(
      builder, op, &align_corners, &half_pixel_centers));
  return Expected<void>();
}

LiteRtStatus LeakyReluOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflLeakyRelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetLeakyReluAlphaOption(op, &alpha));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> LeakyReluOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildLeakyReluOpOption(builder, op, &alpha));
  return Expected<void>();
}

LiteRtStatus SpaceToDepthOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSpaceToDepth) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSpaceToDepthBlockSizeOption(op, &block_size));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> SpaceToDepthOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSpaceToDepthOpOption(builder, op, &block_size));
  return Expected<void>();
}

LiteRtStatus DepthToSpaceOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflDepthToSpace) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthToSpaceBlockSizeOption(op, &block_size));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> DepthToSpaceOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildDepthToSpaceOpOption(builder, op, &block_size));
  return Expected<void>();
}

LiteRtStatus ResizeNearestNeighborOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflResizeNearestNeighbor) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeNearestNeighborAlignCornersOption(op, &align_corners));
  LITERT_RETURN_IF_ERROR(LiteRtGetResizeNearestNeighborHalfPixelCenterOption(
      op, &half_pixel_centers));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> ResizeNearestNeighborOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildResizeNearestNeighborOpOption(
      builder, op, &align_corners, &half_pixel_centers));
  return Expected<void>();
}

LiteRtStatus CumSumOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflCumsum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetCumsumExclusiveOption(op, &exclusive));
  LITERT_RETURN_IF_ERROR(LiteRtGetCumsumReverseOption(op, &reverse));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> CumSumOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildCumsumOpOption(builder, op, &exclusive, &reverse));
  return Expected<void>();
}

LiteRtStatus GeluOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflGelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetGeluApproximateOption(op, &approximate));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> GeluOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildGeluOpOption(builder, op, &approximate));
  return Expected<void>();
}

LiteRtStatus MirrorPadOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMirrorPad) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetMirrorPadModeOption(op, &mode));
  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> MirrorPadOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildMirrorPadOpOption(builder, op, &mode));
  return Expected<void>();
}

LiteRtStatus SqueezeOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSqueeze) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const int32_t* squeeze_dims_data;
  int32_t num_squeeze_dims;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetSqueezeDimsOption(op, &squeeze_dims_data, &num_squeeze_dims));
  squeeze_dims.assign(squeeze_dims_data, squeeze_dims_data + num_squeeze_dims);

  this->op = op;

  return kLiteRtStatusOk;
}

Expected<void> SqueezeOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildSqueezeOpOption(
      builder, op, squeeze_dims.data(), squeeze_dims.size()));
  return Expected<void>();
}
}  // namespace litert
