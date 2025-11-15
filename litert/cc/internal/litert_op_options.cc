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
}  // namespace litert
