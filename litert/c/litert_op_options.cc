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

#include "litert/c/litert_op_options.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "tflite/schema/schema_generated.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Op Options
//

LiteRtStatus LiteRtGetAddFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorNotFound;
  }
  *fused_activation = opts.AsAddOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAdjXOption(LiteRtOp op, bool* adj_x) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_x = opts.AsBatchMatMulOptions()->adj_x;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAdjYOption(LiteRtOp op, bool* adj_y) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_y = opts.AsBatchMatMulOptions()->adj_y;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      opts.AsBatchMatMulOptions()->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConcatenationFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsConcatenationOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConcatenationAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsConcatenationOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDivFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsDivOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsFullyConnectedOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedKeepNumDimsOption(LiteRtOp op,
                                                      bool* keep_num_dims) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *keep_num_dims = opts.AsFullyConnectedOptions()->keep_num_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetQuantizedBiasTypeOption(
    LiteRtOp op, uint32_t* quantized_bias_type) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *quantized_bias_type = opts.AsFullyConnectedOptions()->quantized_bias_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      opts.AsFullyConnectedOptions()->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedWeightsFormatOption(
    LiteRtOp op, uint32_t* weights_format) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *weights_format = opts.AsFullyConnectedOptions()->weights_format;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMulFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsMulOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSoftmaxBetaOption(LiteRtOp op, float* beta) {
  if (op->OpCode() != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *beta = opts.AsSoftmaxOptions()->beta;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceBeginMaskOption(LiteRtOp op,
                                                  int32_t* begin_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *begin_mask = opts.AsStridedSliceOptions()->begin_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceEndMaskOption(LiteRtOp op,
                                                int32_t* end_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *end_mask = opts.AsStridedSliceOptions()->end_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceEllipsisMaskOption(LiteRtOp op,
                                                     int32_t* ellipsis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *ellipsis_mask = opts.AsStridedSliceOptions()->ellipsis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceNewAxisMaskOption(LiteRtOp op,
                                                    int32_t* new_axis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *new_axis_mask = opts.AsStridedSliceOptions()->new_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceShrinkAxisMaskOption(
    LiteRtOp op, int32_t* shrink_axis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *shrink_axis_mask = opts.AsStridedSliceOptions()->shrink_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceOffsetOption(LiteRtOp op, bool* offset) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *offset = opts.AsStridedSliceOptions()->offset;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsSubOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReshapeNewShapeOption(LiteRtOp op,
                                            const int32_t** new_shape,
                                            int32_t* new_shape_size) {
  if (op->OpCode() != kLiteRtOpCodeTflReshape) {
    LITERT_LOG(LITERT_WARNING, "Expected Reshape op, but got: %d",
               op->OpCode());
    return kLiteRtStatusErrorInvalidArgument;
  }
  // The new shape is stored as the second input to the OP as a i32 tensor, as
  // per 'lite/ir/tfl_ops.td' 'TFL_ReshapeOp' definition.
  if (op->NumInputs() < 2) {
    LITERT_LOG(LITERT_WARNING,
               "Expected at least 2 inputs for Reshape op, but got: %d",
               op->NumInputs());
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtTensor new_shape_tensor = op->Inputs()[1];
  LiteRtRankedTensorType ranked_tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetRankedTensorType(new_shape_tensor, &ranked_tensor_type));
  if (ranked_tensor_type.element_type != kLiteRtElementTypeInt32) {
    LITERT_LOG(LITERT_WARNING,
               "Expected int32 element type for new shape tensor, but got: %d",
               ranked_tensor_type.element_type);
    return kLiteRtStatusErrorInvalidArgument;
  }
  size_t num_elements = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetNumLayoutElements(&(ranked_tensor_type.layout), &num_elements));
  if (num_elements <= 0) {
    LITERT_LOG(LITERT_WARNING,
               "Expected positive number of elements for new shape tensor, but "
               "got: %zu",
               num_elements);
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (new_shape_tensor->Weights().Buffer().Size() <= 0) {
    LITERT_LOG(
        LITERT_WARNING,
        "Expected positive size for new shape tensor buffer, but got: %zu",
        new_shape_tensor->Weights().Buffer().Size());
    return kLiteRtStatusErrorInvalidArgument;
  }

  *new_shape = reinterpret_cast<const int32_t*>(
      new_shape_tensor->Weights().Buffer().Data());
  *new_shape_size = num_elements;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSumKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Sum OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReduceMaxKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflReduceMax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // ReduceMax OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReduceMinKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflReduceMin) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // ReduceMin OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReduceAnyKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflReduceAny) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // ReduceAny OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReduceAllKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflReduceAll) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // ReduceAll OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetPackAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflPack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsPackOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetUnpackAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflUnpack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsUnpackOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGatherAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsGatherOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGatherBatchDimsOption(LiteRtOp op, int32_t* batch_dims) {
  if (op->OpCode() != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *batch_dims = opts.AsGatherOptions()->batch_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMeanKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Mean OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSplitNumSplitsOption(LiteRtOp op, int32_t* num_splits) {
  if (op->OpCode() != kLiteRtOpCodeTflSplit) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_splits = opts.AsSplitOptions()->num_splits;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSHLOCompositeOpName(LiteRtOp op, const char** name) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions2(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *name = opts.AsStableHLOCompositeOptions()->name.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex(
    LiteRtOp op, int32_t* subgraph_index) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions2(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *subgraph_index =
      opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSHLOCompositeOpAttributes(LiteRtOp op,
                                                const uint8_t** attributes,
                                                int32_t* attributes_size) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions2(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const tflite::StableHLOCompositeOptionsT* stable_hlo_composite_options =
      opts.AsStableHLOCompositeOptions();
  if (stable_hlo_composite_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *attributes = stable_hlo_composite_options->composite_attributes.data();
  *attributes_size = stable_hlo_composite_options->composite_attributes.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSHLOCompositeOpVersion(LiteRtOp op, int32_t* version) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions2(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const tflite::StableHLOCompositeOptionsT* stable_hlo_composite_options =
      opts.AsStableHLOCompositeOptions();
  if (stable_hlo_composite_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *version = stable_hlo_composite_options->version;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dPaddingOption(LiteRtOp op, uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsConv2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dStrideWOption(LiteRtOp op, int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsConv2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dStrideHOption(LiteRtOp op, int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsConv2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsConv2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dDilationWOption(LiteRtOp op,
                                            int32_t* dilation_w_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_w_factor = opts.AsConv2DOptions()->dilation_w_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dDilationHOption(LiteRtOp op,
                                            int32_t* dilation_h_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_h_factor = opts.AsConv2DOptions()->dilation_h_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dPaddingOption(LiteRtOp op, uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsConv3DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dStrideDOption(LiteRtOp op, int32_t* stride_d) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_d = opts.AsConv3DOptions()->stride_d;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dStrideWOption(LiteRtOp op, int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsConv3DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dStrideHOption(LiteRtOp op, int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsConv3DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsConv3DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dDilationDOption(LiteRtOp op,
                                            int32_t* dilation_d_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_d_factor = opts.AsConv3DOptions()->dilation_d_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dDilationWOption(LiteRtOp op,
                                            int32_t* dilation_w_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_w_factor = opts.AsConv3DOptions()->dilation_w_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv3dDilationHOption(LiteRtOp op,
                                            int32_t* dilation_h_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_h_factor = opts.AsConv3DOptions()->dilation_h_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTransposeConvPaddingOption(LiteRtOp op,
                                                 uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsTransposeConvOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTransposeConvStrideWOption(LiteRtOp op,
                                                 int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsTransposeConvOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTransposeConvStrideHOption(LiteRtOp op,
                                                 int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsTransposeConvOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTransposeConvFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsTransposeConvOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dPaddingOption(LiteRtOp op,
                                                   uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsDepthwiseConv2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dStrideWOption(LiteRtOp op,
                                                   int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsDepthwiseConv2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dStrideHOption(LiteRtOp op,
                                                   int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsDepthwiseConv2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dDepthMultiplierOption(
    LiteRtOp op, int32_t* depth_multiplier) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *depth_multiplier = opts.AsDepthwiseConv2DOptions()->depth_multiplier;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsDepthwiseConv2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dDilationWOption(
    LiteRtOp op, int32_t* dilation_w_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_w_factor = opts.AsDepthwiseConv2DOptions()->dilation_w_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dDilationHOptions(
    LiteRtOp op, int32_t* dilation_h_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_h_factor = opts.AsDepthwiseConv2DOptions()->dilation_h_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dOptions(LiteRtOp op, int8_t* padding,
                                           int32_t* stride_w, int32_t* stride_h,
                                           int32_t* filter_width,
                                           int32_t* filter_height,
                                           int8_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto* options = opts.AsPool2DOptions();
  *padding = options->padding;
  *stride_w = options->stride_w;
  *stride_h = options->stride_h;
  *filter_width = options->filter_width;
  *filter_height = options->filter_height;
  *fused_activation_function = options->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dPaddingOption(LiteRtOp op,
                                                 uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsPool2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dStrideWOption(LiteRtOp op,
                                                 int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsPool2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dStrideHOption(LiteRtOp op,
                                                 int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsPool2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dFilterWidthOption(LiteRtOp op,
                                                     int32_t* filter_width) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *filter_width = opts.AsPool2DOptions()->filter_width;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dFilterHeightOption(LiteRtOp op,
                                                      int32_t* filter_height) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *filter_height = opts.AsPool2DOptions()->filter_height;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsPool2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMaxPool2dPaddingOption(LiteRtOp op, uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsPool2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMaxPool2dStrideWOption(LiteRtOp op, int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsPool2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMaxPool2dStrideHOption(LiteRtOp op, int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsPool2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMaxPool2dFilterWidthOption(LiteRtOp op,
                                                 int32_t* filter_width) {
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *filter_width = opts.AsPool2DOptions()->filter_width;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMaxPool2dFilterHeightOption(LiteRtOp op,
                                                  int32_t* filter_height) {
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *filter_height = opts.AsPool2DOptions()->filter_height;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMaxPool2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsPool2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetResizeBilinearAlignCornersOption(LiteRtOp op,
                                                       bool* align_corners) {
  if (op->OpCode() != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *align_corners = opts.AsResizeBilinearOptions()->align_corners;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetResizeBilinearHalfPixelCenterOption(
    LiteRtOp op, bool* half_pixel_centers) {
  if (op->OpCode() != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *half_pixel_centers = opts.AsResizeBilinearOptions()->half_pixel_centers;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetLeakyReluAlphaOption(LiteRtOp op, float* alpha) {
  if (op->OpCode() != kLiteRtOpCodeTflLeakyRelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *alpha = opts.AsLeakyReluOptions()->alpha;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthToSpaceBlockSizeOption(LiteRtOp op,
                                                  int32_t* block_size) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthToSpace) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *block_size = opts.AsDepthToSpaceOptions()->block_size;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSpaceToDepthBlockSizeOption(LiteRtOp op,
                                                  int32_t* block_size) {
  if (op->OpCode() != kLiteRtOpCodeTflSpaceToDepth) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *block_size = opts.AsSpaceToDepthOptions()->block_size;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetResizeNearestNeighborAlignCornersOption(
    LiteRtOp op, bool* align_corners) {
  if (op->OpCode() != kLiteRtOpCodeTflResizeNearestNeighbor) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *align_corners = opts.AsResizeNearestNeighborOptions()->align_corners;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetResizeNearestNeighborHalfPixelCenterOption(
    LiteRtOp op, bool* half_pixel_centers) {
  if (op->OpCode() != kLiteRtOpCodeTflResizeNearestNeighbor) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *half_pixel_centers =
      opts.AsResizeNearestNeighborOptions()->half_pixel_centers;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCumsumExclusiveOption(LiteRtOp op, bool* exclusive) {
  if (op->OpCode() != kLiteRtOpCodeTflCumsum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *exclusive = opts.AsCumsumOptions()->exclusive;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCumsumReverseOption(LiteRtOp op, bool* reverse) {
  if (op->OpCode() != kLiteRtOpCodeTflCumsum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *reverse = opts.AsCumsumOptions()->reverse;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGeluApproximateOption(LiteRtOp op, bool* approximate) {
  if (op->OpCode() != kLiteRtOpCodeTflGelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *approximate = opts.AsGeluOptions()->approximate;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMirrorPadModeOption(LiteRtOp op, uint32_t* mode) {
  if (op->OpCode() != kLiteRtOpCodeTflMirrorPad) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *mode = opts.AsMirrorPadOptions()->mode;
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
