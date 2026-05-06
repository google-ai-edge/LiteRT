// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/samsung/compiler/create_model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "common-types.h"  // from @exynos_ai_litecore
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/samsung/ai_litecore_manager.h"
#include "litert/vendors/samsung/compiler/builders/argmax_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/batch_matmul_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/broadcastto_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/cast_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/concat_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/conv2d_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/conv3d_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/cumsum_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/dequantize_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/elementwise_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/fully_connected_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/gather_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/gathernd_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/gelu_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/hardswish_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/l2normalization_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/leakyrelu_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/logistic_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"
#include "litert/vendors/samsung/compiler/builders/pad_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/pool2d_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/quantize_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/reduce_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/relu1_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/relu6_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/relu_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/relun1to1_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/reshape_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/resizebilinear_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/resizenearestneighbor_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/reversev2_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/rms_norm_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/select_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/selectv2_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/slice_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/softmax_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/spacetodepth_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/split_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/strided_slice_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/tanh_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/transpose_conv_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/transpose_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

// Util
const char* MapToQuantTypeStr(ElementType element_type) {
  switch (element_type) {
    case ElementType::Int4:
      return "AINT4";
    case ElementType::Int8:
    case ElementType::UInt8:
      return "AINT8";
    case ElementType::Int16:
    case ElementType::UInt16:
      return "AINT16";
    case ElementType::Int32:
      return "AINT32";
    default:
      return "";
  }
}

// GraphCreator
GraphCreator::GraphCreator(AiLiteCoreManager::Ptr ai_lite_core,
                           graph_handler_t handler)
    : ai_lite_core_(ai_lite_core), handler_(handler) {}

LiteRtStatus GraphCreator::CreateTensor(const Tensor& t) {
  if (!t.Get()) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  uint32_t tensor_index = t.TensorIndex();
  if (tensors_map_.find(tensor_index) != tensors_map_.end()) {
    return kLiteRtStatusOk;
  }

  auto ranked_tensor_type = t.RankedTensorType();
  if (!ranked_tensor_type) {
    return static_cast<LiteRtStatus>(ranked_tensor_type.Error().StatusCC());
  }

  auto dimensions = ranked_tensor_type->Layout().Dimensions();
  std::vector<DIM_T> tensor_shape(dimensions.size());
  absl::c_copy(dimensions, tensor_shape.begin());

  if (tensor_shape.empty()) {
    // Shape size of tensor couldn't be zero
    tensor_shape = {1};
  }

  auto element_type = ranked_tensor_type->ElementType();
  auto element_type_mapping = MapToElementTypeStr(element_type);
  if (!element_type_mapping.HasValue()) {
    return static_cast<LiteRtStatus>(element_type_mapping.Error().StatusCC());
  }

  const char* layout_rep = tensor_shape.size() == 4 ? "NHWC" : "UNDEFINED";
  TENSOR_ID_T tensor_id;
  LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(ai_lite_core_->Api().DefineTensor(
      handler_, &tensor_id, t.Name().data(), tensor_shape.data(),
      tensor_shape.size(), element_type_mapping.Value(), layout_rep));
  tensors_map_[tensor_index] = tensor_id;

  if (t.HasQuantization()) {
    LITERT_RETURN_IF_ERROR(CreateQParam(t));
  }
  if (t.HasWeights()) {
    auto weight = t.Weights().Bytes();
    LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(
        ai_lite_core_->Api().SetTensorData(handler_, tensor_id, weight.data(),
                                           weight.size()));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus GraphCreator::CreateOpNode(const OpWrapper& op_wrapper) {
  NODE_ID_T op_id;

  std::vector<TENSOR_ID_T> input_indices;
  std::vector<TENSOR_ID_T> output_indices;
  for (const auto& input : op_wrapper.GetInputs()) {
    LITERT_RETURN_IF_ERROR(CreateTensor(input));
    input_indices.push_back(tensors_map_.at(input.TensorIndex()));
  }

  for (const auto& output : op_wrapper.GetOutputs()) {
    LITERT_RETURN_IF_ERROR(CreateTensor(output));
    output_indices.push_back(tensors_map_.at(output.TensorIndex()));
  }
  LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(ai_lite_core_->Api().DefineOp(
      handler_, &op_id, op_wrapper.GetCName(), op_wrapper.GetCType(),
      input_indices.data(), input_indices.size(), output_indices.data(),
      output_indices.size()));

  for (int32_t index = 0; index < op_wrapper.GetNumOfParams(); index++) {
    const auto& param = op_wrapper.GetParamWithIndex(index);
    LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(ai_lite_core_->Api().AddOpParam(
        handler_, op_id, param.GetKey().c_str(), param.GetValue()));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus GraphCreator::AddInput(const Tensor& t_input) {
  uint32_t tensor_input_index = t_input.TensorIndex();
  if (tensors_map_.find(tensor_input_index) == tensors_map_.end()) {
    LITERT_RETURN_IF_ERROR(CreateTensor(t_input));
  }
  input_indices_.push_back(tensors_map_.at(tensor_input_index));

  return kLiteRtStatusOk;
}

LiteRtStatus GraphCreator::AddOutput(const Tensor& t_output) {
  uint32_t tensor_output_index = t_output.TensorIndex();
  if (tensors_map_.find(tensor_output_index) == tensors_map_.end()) {
    LITERT_RETURN_IF_ERROR(CreateTensor(t_output));
  }
  output_indices_.push_back(tensors_map_.at(tensor_output_index));

  return kLiteRtStatusOk;
}

LiteRtStatus GraphCreator::Finish() const {
  LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(
      ai_lite_core_->Api().SetGraphInputs(handler_, input_indices_.data(),
                                          input_indices_.size()));
  LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(
      ai_lite_core_->Api().SetGraphOutputs(handler_, output_indices_.data(),
                                           output_indices_.size()));
  LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(
      ai_lite_core_->Api().FinishGraphBuild(handler_));

  return kLiteRtStatusOk;
}

Expected<std::vector<char>> GraphCreator::Release() const {
  uint8_t* graph_ptr;
  uint64_t num_bytes;
  if (ai_lite_core_->Api().Serialize(handler_, &graph_ptr, &num_bytes) !=
      GraphWrapperReturn::SUCCESS) {
    return Error(litert::Status::kErrorRuntimeFailure,
                 "Fail to serialize graph");
  }

  auto g_buffer = std::vector<char>(num_bytes / sizeof(char));
  memcpy(g_buffer.data(), graph_ptr, num_bytes);

  return g_buffer;
}

LiteRtStatus GraphCreator::CreateQParam(const Tensor& t) {
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
  if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
    auto quant_info = t.PerTensorQuantization();
    scales = {quant_info.scale};
    zero_points = {static_cast<int32_t>(quant_info.zero_point)};
  } else if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
    auto quant_info = t.PerChannelQuantization();
    auto num_channels = quant_info.num_channels;
    scales.resize(num_channels);
    zero_points.resize(num_channels);
    std::copy(quant_info.scales, quant_info.scales + num_channels,
              scales.begin());
    std::copy(quant_info.zero_points, quant_info.zero_points + num_channels,
              zero_points.begin());
    LITERT_LOG(LITERT_INFO, "quantized_dimension: %d",
               quant_info.quantized_dimension);
  } else {
    LITERT_LOG(LITERT_ERROR, "Not Supported QTypeId: %d",
               static_cast<int>(t.QTypeId()));
  }

  if (scales.empty() || zero_points.empty()) {
    LITERT_LOG(LITERT_INFO, "Fail to get scale and zero point.");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  ParamWrapper scale_info = {.size = static_cast<uint32_t>(scales.size()),
                             .data = scales.data(),
                             .is_scalar = false,
                             .type = ScalarType::FLOAT32};
  ParamWrapper zero_point_info = {
      .size = static_cast<uint32_t>(zero_points.size()),
      .data = zero_points.data(),
      .is_scalar = false,
      .type = ScalarType::INT32};

  auto ranked_tensor_type = t.RankedTensorType();
  auto element_type = ranked_tensor_type->ElementType();
  LITERT_RETURN_STATUS_IF_AI_LITECORE_NOT_OK(
      ai_lite_core_->Api().SetTensorQParam(
          handler_, tensors_map_.at(t.TensorIndex()),
          MapToQuantTypeStr(element_type), scale_info, zero_point_info));

  return kLiteRtStatusOk;
}

// Create Model Main
Expected<std::vector<char>> CreateModel(AiLiteCoreManager::Ptr ai_lite_core,
                                        const Subgraph& partition) {
  auto expected_graph_handler = ai_lite_core->CreateGraphHandler();
  if (!expected_graph_handler.HasValue()) {
    return expected_graph_handler.Error();
  }
  auto graph_handler = std::move(expected_graph_handler.Value());

  GraphCreator graph_crt(ai_lite_core, graph_handler.get());
  for (const Tensor& input : partition.Inputs()) {
    LITERT_RETURN_IF_ERROR(graph_crt.AddInput(input));
  }
  for (const Tensor& output : partition.Outputs()) {
    LITERT_RETURN_IF_ERROR(graph_crt.AddOutput(output));
  }

  auto ops = partition.Ops();
  for (int op_idx = 0; op_idx < ops.size(); ++op_idx) {
    const auto& op = ops[op_idx];
    Expected<OpWrapper> op_wrapper =
        Error(litert::Status::kErrorInvalidArgument, "Invalid op wrapper");
    switch (op.Code()) {
      case kLiteRtOpCodeTflAbs:
        op_wrapper = BuildAbsOp(op);
        break;
      case kLiteRtOpCodeTflAdd:
        op_wrapper = BuildAddOp(op);
        break;
      case kLiteRtOpCodeTflArgMax:
        op_wrapper = BuildArgMaxOp(op);
        break;
      case kLiteRtOpCodeTflAveragePool2d:
        op_wrapper = BuildAvgPool2dOp(op);
        break;
      case kLiteRtOpCodeTflBatchMatmul:
        op_wrapper = BuildBatchMatMulOp(op);
        break;
      case kLiteRtOpCodeTflBroadcastTo:
        op_wrapper = BuildBroadcastToOp(op);
        break;
      case kLiteRtOpCodeTflCast:
        op_wrapper = BuildCastOp(op);
        break;
      case kLiteRtOpCodeTflCeil:
        op_wrapper = BuildCeilOp(op);
        break;
      case kLiteRtOpCodeTflConcatenation:
        op_wrapper = BuildConcatOp(op);
        break;
      case kLiteRtOpCodeTflConv2d:
        op_wrapper = BuildConv2dOp(op);
        break;
      case kLiteRtOpCodeTflConv3d:
        op_wrapper = BuildConv3dOp(op);
        break;
      case kLiteRtOpCodeTflCos:
        op_wrapper = BuildCosOp(op);
        break;
      case kLiteRtOpCodeTflCumsum:
        op_wrapper = BuildCumsumOp(op);
        break;
      case kLiteRtOpCodeTflDepthwiseConv2d:
        op_wrapper = BuildDepthwiseConv2dOp(op);
        break;
      case kLiteRtOpCodeTflDequantize:
        op_wrapper = BuildDequantizeOp(op);
        break;
      case kLiteRtOpCodeTflDiv:
        op_wrapper = BuildDivOp(op);
        break;
      case kLiteRtOpCodeTflEmbeddingLookup:
        op_wrapper = BuildEmbeddingLookupOp(op);
        break;
      case kLiteRtOpCodeTflEqual:
        op_wrapper = BuildEqualOp(op);
        break;
      case kLiteRtOpCodeTflExp:
        op_wrapper = BuildExpOp(op);
        break;
      case kLiteRtOpCodeTflFloor:
        op_wrapper = BuildFloorOp(op);
        break;
      case kLiteRtOpCodeTflFloorDiv:
        op_wrapper = BuildFloorDivOp(op);
        break;
      case kLiteRtOpCodeTflFullyConnected:
        op_wrapper = BuildFullyConnectedOp(op);
        break;
      case kLiteRtOpCodeTflGather:
        op_wrapper = BuildGatherOp(op);
        break;
      case kLiteRtOpCodeTflGatherNd:
        op_wrapper = BuildGatherNdOp(op);
        break;
      case kLiteRtOpCodeTflGelu:
        op_wrapper = BuildGeluOp(op);
        break;
      case kLiteRtOpCodeTflGreater:
        op_wrapper = BuildGreaterOp(op);
        break;
      case kLiteRtOpCodeTflGreaterEqual:
        op_wrapper = BuildGreaterEqualOp(op);
        break;
      case kLiteRtOpCodeTflHardSwish:
        op_wrapper = BuildHardSwishOp(op);
        break;
      case kLiteRtOpCodeTflL2Normalization:
        op_wrapper = BuildL2NormalizationOp(op);
        break;
      case kLiteRtOpCodeTflLeakyRelu:
        op_wrapper = BuildLeakyReluOp(op);
        break;
      case kLiteRtOpCodeTflLess:
        op_wrapper = BuildLessOp(op);
        break;
      case kLiteRtOpCodeTflLog:
        op_wrapper = BuildLogOp(op);
        break;
      case kLiteRtOpCodeTflLogicalAnd:
        op_wrapper = BuildLogicalAndOp(op);
        break;
      case kLiteRtOpCodeTflLogistic:
        op_wrapper = BuildLogisticOp(op);
        break;
      case kLiteRtOpCodeTflMaximum:
        op_wrapper = BuildMaxOp(op);
        break;
      case kLiteRtOpCodeTflMinimum:
        op_wrapper = BuildMinOp(op);
        break;
      case kLiteRtOpCodeTflMirrorPad:
        op_wrapper = BuildMirrorPadOp(op);
        break;
      case kLiteRtOpCodeTflMaxPool2d:
        op_wrapper = BuildMaxPool2dOp(op);
        break;
      case kLiteRtOpCodeTflMean:
        op_wrapper = BuildReduceMeanOp(op);
        break;
      case kLiteRtOpCodeTflMul:
        op_wrapper = BuildMulOp(op);
        break;
      case kLiteRtOpCodeTflNotEqual:
        op_wrapper = BuildNotEqualOp(op);
        break;
      case kLiteRtOpCodeTflPad:
      case kLiteRtOpCodeTflPadv2:
        op_wrapper = BuildPadOp(op);
        break;
      case kLiteRtOpCodeTflPow:
        op_wrapper = BuildPowOp(op);
        break;
      case kLiteRtOpCodeTflQuantize:
        op_wrapper = BuildQuantizeOp(op);
        break;
      case kLiteRtOpCodeTflRelu:
        op_wrapper = BuildReLUOp(op);
        break;
      case kLiteRtOpCodeTflRelu0To1:
        op_wrapper = BuildRelu1Op(op);
        break;
      case kLiteRtOpCodeTflReluN1To1:
        op_wrapper = BuildReluN1To1(op);
        break;
      case kLiteRtOpCodeTflRelu6:
        op_wrapper = BuildRelu6Op(op);
        break;
      case kLiteRtOpCodeTflReshape:
        op_wrapper = BuildReshapeOp(op);
        break;
      case kLiteRtOpCodeTflReduceMax:
        op_wrapper = BuildReduceMaxOp(op);
        break;
      case kLiteRtOpCodeTflReduceMin:
        op_wrapper = BuildReduceMinOp(op);
        break;
      case kLiteRtOpCodeTflResizeBilinear:
        op_wrapper = BuildResizeBilinearOp(op);
        break;
      case kLiteRtOpCodeTflResizeNearestNeighbor:
        op_wrapper = BuildResizeNearestNeighborOp(op);
        break;
      case kLiteRtOpCodeTflReverseV2:
        op_wrapper = BuildReverseV2Op(op);
        break;
      case kLiteRtOpCodeTflRsqrt:
        op_wrapper = BuildRsqrtOp(op);
        break;
      case kLiteRtOpCodeTflSelect:
        op_wrapper = BuildSelectOp(op);
        break;
      case kLiteRtOpCodeTflSelectV2:
        op_wrapper = BuildSelectV2Op(op);
        break;
      case kLiteRtOpCodeTflSin:
        op_wrapper = BuildSinOp(op);
        break;
      case kLiteRtOpCodeTflSlice:
        op_wrapper = BuildSliceOp(op);
        break;
      case kLiteRtOpCodeTflSoftmax:
        op_wrapper = BuildSoftmaxOp(op);
        break;
      case kLiteRtOpCodeTflSpaceToDepth:
        op_wrapper = BuildSpaceToDepthOp(op);
        break;
      case kLiteRtOpCodeTflSplit:
        op_wrapper = BuildSplitOp(op);
        break;
      case kLiteRtOpCodeTflSqrt:
        op_wrapper = BuildSqrtOp(op);
        break;
      case kLiteRtOpCodeTflSquaredDifference:
        op_wrapper = BuildSquaredDifferenceOp(op);
        break;
      case kLiteRtOpCodeTflSub:
        op_wrapper = BuildSubOp(op);
        break;
      case kLiteRtOpCodeTflSum:
        op_wrapper = BuildReduceSumOp(op);
        break;
      case kLiteRtOpCodeTflStridedSlice:
        op_wrapper = BuildStridedSliceOp(op);
        break;
      case kLiteRtOpCodeTflTanh:
        op_wrapper = BuildTanhOp(op);
        break;
      case kLiteRtOpCodeTflTranspose:
        op_wrapper = BuildTransposeOp(op);
        break;
      case kLiteRtOpCodeTflTransposeConv:
        op_wrapper = BuildTransposeConvOp(op);
        break;
      case kLiteRtOpCodeShloComposite: {
        auto composite_opt = GetOptionsAs<CompositeOptions>(op.Get());
        if (composite_opt &&
            composite_opt->name == CompositeOptions::kRmsNorm) {
          op_wrapper = BuildRmsNormOp(op);
        } else {
          LITERT_LOG(LITERT_ERROR,
                     "Unsupported composite operator : ", composite_opt->name);
        }
        break;
      }
      default:
        LITERT_LOG(LITERT_ERROR, "Unsupported op: %d", op.Code());
    }
    if (!op_wrapper) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Fail to parse op's options.");
    }
    if (auto status = graph_crt.CreateOpNode(op_wrapper.Value());
        status != kLiteRtStatusOk) {
      return Error(static_cast<litert::Status>(status),
                   absl::StrFormat("Fail to build op (index:%d)", op_idx));
    }
  }
  if (auto status = graph_crt.Finish(); status != kLiteRtStatusOk) {
    return Error(static_cast<litert::Status>(status),
                 "Fail to build graph for samsung backend.");
  }

  return graph_crt.Release();
}

}  // namespace litert::samsung
