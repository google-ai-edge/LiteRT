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
#include "litert/vendors/qualcomm/compiler/qnn_compose_graph.h"

#include <alloca.h>
#include <stdbool.h>
#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/cc/namespace_heuristics.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "litert/vendors/qualcomm/core/builders/arg_min_max_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/broadcast_to_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/conv3d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/cumsum_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/depthwise_conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/dynamic_update_slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder_htp.h"
#include "litert/vendors/qualcomm/core/builders/gather_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/gathernd_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/gelu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/group_norm_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/l2_norm_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/leaky_relu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/log_softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/logistic_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pad_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pool2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/prelu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reduce_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu6_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_0to1_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_n1to1_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/resize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reverse_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/rms_norm_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/scatter_nd_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/spatial_transform_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/strided_slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/tanh_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/tile_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/topk_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_conv_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/dump/dump_graph.h"
#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/model_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
namespace litert::qnn {
namespace {
static const char* kLiteRtStr = "litert";
}

LiteRtStatus ConvertPaddingType(const uint32_t litert_padding,
                                ::qnn::PaddingType& qnn_padding) {
  switch (litert_padding) {
    case 0: {
      qnn_padding = ::qnn::PaddingType::Same;
      break;
    }
    case 1: {
      qnn_padding = ::qnn::PaddingType::Valid;
      break;
    }
    default: {
      return kLiteRtStatusErrorUnsupported;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertDataType(const litert::ElementType litert_type,
                             const bool is_quantized,
                             Qnn_DataType_t& qnn_type) {
  qnn_type = QNN_DATATYPE_UNDEFINED;
  switch (litert_type) {
    case litert::ElementType::Bool:
      qnn_type = QNN_DATATYPE_BOOL_8;
      break;
    case litert::ElementType::Int4:
      qnn_type = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    case litert::ElementType::Int8:
      qnn_type =
          is_quantized ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_INT_8;
      break;
    case litert::ElementType::Int16:
      qnn_type =
          is_quantized ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_INT_16;
      break;
    case litert::ElementType::Int32:
      qnn_type =
          is_quantized ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_INT_32;
      break;
    case litert::ElementType::Int64:
      qnn_type = QNN_DATATYPE_INT_64;
      break;
    case litert::ElementType::UInt8:
      qnn_type =
          is_quantized ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_UINT_8;
      break;
    case litert::ElementType::UInt16:
      qnn_type =
          is_quantized ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UINT_16;
      break;
    case litert::ElementType::UInt32:
      qnn_type =
          is_quantized ? QNN_DATATYPE_UFIXED_POINT_32 : QNN_DATATYPE_UINT_32;
      break;
    case litert::ElementType::UInt64:
      qnn_type = QNN_DATATYPE_UINT_64;
      break;
    case litert::ElementType::Float16:
      qnn_type = QNN_DATATYPE_FLOAT_16;
      break;
    case litert::ElementType::Float32:
      qnn_type = QNN_DATATYPE_FLOAT_32;
      break;
    case litert::ElementType::Float64:
      qnn_type = QNN_DATATYPE_FLOAT_64;
      break;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertTensor(const litert::Tensor& litert_tensor,
                           ::qnn::TensorPool& tensor_pool,
                           ::qnn::TensorWrapper*& tensor_wrapper,
                           const absl::flat_hash_set<std::int32_t>& ids_to_dump,
                           bool is_tensor_read_and_write) {
  tensor_wrapper = nullptr;

  if (litert_tensor.TypeId() != kLiteRtRankedTensorType) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto ranked_tensor_type = litert_tensor.RankedTensorType();
  if (!ranked_tensor_type) {
    LITERT_LOG(LITERT_ERROR, "%s", ranked_tensor_type.Error().Message().data());
    return ranked_tensor_type.Error().Status();
  }

  Qnn_DataType_t qnn_data_type;
  LITERT_RETURN_IF_ERROR(ConvertDataType(ranked_tensor_type->ElementType(),
                                         litert_tensor.HasQuantization(),
                                         qnn_data_type));

  std::vector<std::uint32_t> dimentions;
  const auto litert_layout = ranked_tensor_type->Layout();
  if (litert_layout.Rank() == 0) {
    dimentions.resize(1, 1);
  } else {
    dimentions.resize(litert_layout.Rank());
    for (size_t i = 0; i < dimentions.size(); ++i) {
      // TODO(jiunkaiy): Integrate QNN dynamic dimension.
      // If any dimension sizes are unknown, they are indicated with -1.
      if (litert_layout.Dimensions()[i] == -1) {
        dimentions[i] = 1;
      } else {
        dimentions[i] = litert_layout.Dimensions()[i];
      }
    }
  }

  ::qnn::QuantizeParamsWrapperVariant quantize_params;
  switch (litert_tensor.QTypeId()) {
    case kLiteRtQuantizationPerTensor: {
      const auto per_tensor_quant = litert_tensor.PerTensorQuantization();
      if (ranked_tensor_type->ElementType() == litert::ElementType::Int4) {
        quantize_params.emplace<::qnn::BwScaleOffsetQuantizeParamsWrapper>(
            ::qnn::kQuantBitWidth4, per_tensor_quant.scale,
            per_tensor_quant.zero_point);
      } else {
        quantize_params.emplace<::qnn::ScaleOffsetQuantizeParamsWrapper>(
            per_tensor_quant.scale, per_tensor_quant.zero_point);
      }
      break;
    }
    case kLiteRtQuantizationPerChannel: {
      const auto per_channel_quant = litert_tensor.PerChannelQuantization();
      // convert zero points from std::int64_t to std::int32_t
      std::vector<std::int32_t> zero_points(per_channel_quant.num_channels);
      for (size_t i = 0; i < zero_points.size(); ++i) {
        zero_points[i] = per_channel_quant.zero_points[i];
      }
      if (ranked_tensor_type->ElementType() == litert::ElementType::Int4) {
        quantize_params.emplace<::qnn::BwAxisScaleOffsetQuantizeParamsWrapper>(
            ::qnn::kQuantBitWidth4, per_channel_quant.quantized_dimension,
            absl::Span<const float>{
                per_channel_quant.scales,
                static_cast<size_t>(per_channel_quant.num_channels)},
            absl::Span<const std::int32_t>{zero_points.data(),
                                           zero_points.size()});
      } else {
        quantize_params.emplace<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
            per_channel_quant.quantized_dimension,
            absl::Span<const float>{
                per_channel_quant.scales,
                static_cast<size_t>(per_channel_quant.num_channels)},
            absl::Span<const std::int32_t>{zero_points.data(),
                                           zero_points.size()});
      }
      break;
    }
    case kLiteRtQuantizationBlockWise: {
      LITERT_LOG(LITERT_ERROR, "Unsupported quantization type.");
      return kLiteRtStatusErrorInvalidArgument;
    }
    case kLiteRtQuantizationNone:
    default:
      break;
  }

  uint32_t tensor_index = litert_tensor.TensorIndex();
  auto litert_suffix =
      "_" + std::string(kLiteRtStr) + "_" + std::to_string(tensor_index);
  if (litert_tensor.IsSubgraphInput()) {
    auto& res = tensor_pool.CreateInputTensorWithSuffix(
        qnn_data_type, quantize_params, dimentions, litert_suffix);
    tensor_wrapper = &res;
  } else if (litert_tensor.Uses().empty() || is_tensor_read_and_write) {
    auto& res = tensor_pool.CreateOutpuTensorWithSuffix(
        qnn_data_type, quantize_params, dimentions, litert_suffix);
    tensor_wrapper = &res;
  } else if (litert_tensor.IsConstant()) {
    LITERT_RETURN_IF_ERROR(
        litert_tensor.HasWeights(),
        ErrorStatusBuilder(kLiteRtStatusErrorInvalidLegalization))
        << "Empty weights for constant tensor.";
    auto& res = tensor_pool.CreateStaticTensorWithSuffix(
        qnn_data_type, quantize_params, dimentions, litert_suffix,
        litert_tensor.Weights().Bytes().size(),
        reinterpret_cast<const void*>(litert_tensor.Weights().Bytes().data()),
        false);
    tensor_wrapper = &res;
  } else {
    auto& res = tensor_pool.CreateNativeTensorWithSuffix(
        qnn_data_type, quantize_params, dimentions, litert_suffix);
    // -1 in ids_to_dump will dump all tensors
    if (ids_to_dump.count(-1) > 0 || ids_to_dump.count(tensor_index) > 0) {
      LITERT_LOG(LITERT_INFO, "LiteRT tensor index: %d is dumped",
                 tensor_index);
      res.MarkDump();
    }
    tensor_wrapper = &res;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertOp(const bool use_htp_preferences,
                       const litert::Op& litert_op,
                       ::qnn::ModelWrapper& model_wrapper,
                       std::vector<::qnn::TensorWrapperRef>& input_tensors,
                       std::vector<::qnn::TensorWrapperRef>& output_tensors,
                       std::string_view prefix, std::string_view suffix) {
  switch (litert_op.Code()) {
    case LiteRtOpCode::kLiteRtOpCodeTflCast: {
      model_wrapper.AddOps(::qnn::BuildCastOp(model_wrapper.GetTensorPool(),
                                              input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflConcatenation: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConcatenationAxisOption(litert_op.Get(), &axis));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops =
          ::qnn::BuildConcatenationOp(model_wrapper.GetTensorPool(),
                                      input_tensors, {activation_input}, axis);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAdd: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetAddFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildElementwiseAddOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input});
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogicalAnd: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseAndOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflBroadcastTo: {
      model_wrapper.AddOps(
          ::qnn::BuildBroadcastToOp(model_wrapper.GetTensorPool(),
                                    input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflCeil: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseCeilOp(model_wrapper.GetTensorPool(),
                                        input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflCos: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseCosOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDiv: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetDivFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildElementwiseDivOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input});
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGreater: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseGreaterOp(model_wrapper.GetTensorPool(),
                                           input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLess: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseLessOp(model_wrapper.GetTensorPool(),
                                        input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMul: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetMulFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildElementwiseMulOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input});
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRsqrt: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseRsqrtOp(model_wrapper.GetTensorPool(),
                                         input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSqrt: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseSqrtOp(model_wrapper.GetTensorPool(),
                                        input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSin: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseSinOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSquaredDifference: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseSquaredDifferenceOp(
              model_wrapper.GetTensorPool(), input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSquare: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseSquareOp(model_wrapper.GetTensorPool(),
                                          input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSub: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetSubFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildElementwiseSubOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input});
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMinimum: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseMinimumOp(model_wrapper.GetTensorPool(),
                                           input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMaximum: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseMaximumOp(model_wrapper.GetTensorPool(),
                                           input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflElu: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseEluOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflFloor: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseFloorOp(model_wrapper.GetTensorPool(),
                                         input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflFloorDiv: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseFloorDivOp(model_wrapper.GetTensorPool(),
                                            input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflNotEqual: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseNotEqualOp(model_wrapper.GetTensorPool(),
                                            input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogicalOr: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseOrOp(model_wrapper.GetTensorPool(),
                                      input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflEmbeddingLookup: {
      model_wrapper.AddOps(
          ::qnn::BuildEmbeddingLookupOp(model_wrapper.GetTensorPool(),
                                        input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflFullyConnected: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedFusedActivationOption(
          litert_op.Get(), &fused_activation));
      bool keep_num_dims{};
      LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedKeepNumDimsOption(
          litert_op.Get(), &keep_num_dims));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      std::vector<::qnn::OpWrapper> ops;
      if (use_htp_preferences) {
        ops = ::qnn::BuildFullyConnectedOpHtp(model_wrapper.GetTensorPool(),
                                              input_tensors, {activation_input},
                                              keep_num_dims);
      }
      if (ops.empty()) {
        ops = ::qnn::BuildFullyConnectedOp(model_wrapper.GetTensorPool(),
                                           input_tensors, {activation_input},
                                           keep_num_dims);
      }
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGather: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(litert_op.Get(), &axis));
      int32_t batch_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetGatherBatchDimsOption(litert_op.Get(), &batch_dims));
      model_wrapper.AddOps(
          ::qnn::BuildGatherOp(model_wrapper.GetTensorPool(), input_tensors,
                               output_tensors, axis, batch_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGelu: {
      model_wrapper.AddOps(::qnn::BuildGeluOp(model_wrapper.GetTensorPool(),
                                              input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRelu: {
      model_wrapper.AddOps(::qnn::BuildReluOp(model_wrapper.GetTensorPool(),
                                              input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReluN1To1: {
      model_wrapper.AddOps(
          ::qnn::BuildReluN1To1Op(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRelu0To1: {
      model_wrapper.AddOps(
          ::qnn::BuildRelu0To1Op(model_wrapper.GetTensorPool(), input_tensors,
                                 output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRelu6: {
      model_wrapper.AddOps(::qnn::BuildRelu6Op(model_wrapper.GetTensorPool(),
                                               input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPrelu: {
      model_wrapper.AddOps(::qnn::BuildPreluOp(model_wrapper.GetTensorPool(),
                                               input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogistic: {
      model_wrapper.AddOps(
          ::qnn::BuildLogisticOp(model_wrapper.GetTensorPool(), input_tensors,
                                 output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflBatchMatmul: {
      bool adj_x{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetBatchMatmulAdjXOption(litert_op.Get(), &adj_x));
      bool adj_y{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetBatchMatmulAdjYOption(litert_op.Get(), &adj_y));
      model_wrapper.AddOps(
          ::qnn::BuildMatmulOp(model_wrapper.GetTensorPool(), input_tensors,
                               output_tensors, adj_x, adj_y),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflQuantize: {
      model_wrapper.AddOps(
          ::qnn::BuildQuantizeOp(model_wrapper.GetTensorPool(), input_tensors,
                                 output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDequantize: {
      model_wrapper.AddOps(
          ::qnn::BuildDequantizeOp(model_wrapper.GetTensorPool(), input_tensors,
                                   output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSum: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSumKeepDimsOption(litert_op.Get(), &keep_dims));
      model_wrapper.AddOps(
          ::qnn::BuildReduceSumOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, keep_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMean: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMeanKeepDimsOption(litert_op.Get(), &keep_dims));
      model_wrapper.AddOps(
          ::qnn::BuildReduceMeanOp(model_wrapper.GetTensorPool(), input_tensors,
                                   output_tensors, keep_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReduceMax: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetReduceMaxKeepDimsOption(litert_op.Get(), &keep_dims));
      model_wrapper.AddOps(
          ::qnn::BuildReduceMaxOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, keep_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReduceMin: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetReduceMinKeepDimsOption(litert_op.Get(), &keep_dims));
      model_wrapper.AddOps(
          ::qnn::BuildReduceMinOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, keep_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReduceAll: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetReduceAllKeepDimsOption(litert_op.Get(), &keep_dims));
      model_wrapper.AddOps(
          ::qnn::BuildReduceAllOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, keep_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReduceAny: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetReduceAnyKeepDimsOption(litert_op.Get(), &keep_dims));
      model_wrapper.AddOps(
          ::qnn::BuildReduceAnyOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, keep_dims),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReshape: {
      model_wrapper.AddOps(::qnn::BuildReshapeOp(model_wrapper.GetTensorPool(),
                                                 input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSelect:
    case LiteRtOpCode::kLiteRtOpCodeTflSelectV2: {
      model_wrapper.AddOps(::qnn::BuildSelectOp(model_wrapper.GetTensorPool(),
                                                input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSlice: {
      model_wrapper.AddOps(::qnn::BuildSliceOp(model_wrapper.GetTensorPool(),
                                               input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSoftmax: {
      float beta{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSoftmaxBetaOption(litert_op.Get(), &beta));
      model_wrapper.AddOps(
          ::qnn::BuildSoftmaxOp(model_wrapper.GetTensorPool(), input_tensors,
                                output_tensors, beta),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSplit: {
      int32_t num_splits{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSplitNumSplitsOption(litert_op.Get(), &num_splits));
      model_wrapper.AddOps(
          ::qnn::BuildSplitOp(model_wrapper.GetTensorPool(), input_tensors,
                              output_tensors, num_splits),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTanh: {
      model_wrapper.AddOps(::qnn::BuildTanhOp(model_wrapper.GetTensorPool(),
                                              input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTranspose: {
      model_wrapper.AddOps(
          ::qnn::BuildTransposeOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTile: {
      model_wrapper.AddOps(::qnn::BuildTileOp(model_wrapper.GetTensorPool(),
                                              input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTopkV2: {
      // TODO (Graham): Refactor all OpBuilder to follow QNN master definition
      ::qnn::TensorWrapper k_tensor = input_tensors[1].get();
      if (!k_tensor.IsTensorStatic() || k_tensor.GetTensorNumElements() != 1) {
        QNN_LOG_ERROR(
            "The param 'k' of TopK OP is not static or not 1 element");
        return {};
      }

      std::uint32_t k_data = 0;
      switch (k_tensor.GetDataType()) {
        case QNN_DATATYPE_UINT_32:
          if (auto k = k_tensor.GetTensorData<std::uint32_t>(); k.has_value()) {
            k_data = k.value()[0];
          }
          break;
        case QNN_DATATYPE_INT_32:
          if (auto k = k_tensor.GetTensorData<std::int32_t>(); k.has_value()) {
            k_data = static_cast<std::uint32_t>(k.value()[0]);
          }
          break;
        default:
          QNN_LOG_ERROR("Unsupported data type: %d for k in TopK OP",
                        k_tensor.GetDataType());
          return {};
      }
      model_wrapper.AddOps(
          ::qnn::BuildTopKOp(model_wrapper.GetTensorPool(), input_tensors,
                             output_tensors, k_data),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPack: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(litert_op.Get(), &axis));
      model_wrapper.AddOps(
          ::qnn::BuildPackOp(model_wrapper.GetTensorPool(), input_tensors,
                             output_tensors, axis),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflUnpack: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetUnpackAxisOption(litert_op.Get(), &axis));
      model_wrapper.AddOps(
          ::qnn::BuildUnpackOp(model_wrapper.GetTensorPool(), input_tensors,
                               output_tensors, axis),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDynamicUpdateSlice: {
      model_wrapper.AddOps(
          ::qnn::BuildDynamicUpdateSliceOp(model_wrapper.GetTensorPool(),
                                           input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeShloComposite: {
      auto info = GetOptionsAs<CompositeOptions>(litert_op.Get());
      if (!info) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (info->name == CompositeOptions::kRmsNorm) {
        auto attributes_map = info->attributes_map.value();
        float epsilon = attributes_map["epsilon"].AsFloat();
        model_wrapper.AddOps(
            ::qnn::BuildRmsNormOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, epsilon),
            prefix, suffix);
      }
      if (info->name == CompositeOptions::kGroupNorm) {
        auto attributes_map = info->attributes_map.value();
        float epsilon = attributes_map["epsilon"].AsFloat();
        int num_groups = attributes_map["num_groups"].AsInt32();
        model_wrapper.AddOps(::qnn::BuildGroupNormOp(
                                 model_wrapper.GetTensorPool(), input_tensors,
                                 output_tensors, epsilon, num_groups),
                             prefix, suffix);
      }
      if (info->name == CompositeOptions::kL2Norm) {
        auto attributes_map = info->attributes_map.value();
        float epsilon = attributes_map["epsilon"].AsFloat();
        model_wrapper.AddOps(
            ::qnn::BuildL2NormOp(model_wrapper.GetTensorPool(), input_tensors,
                                 output_tensors, epsilon),
            prefix, suffix);
      }
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflL2Normalization: {
      // TODO(yunandrew): Support custom epsilon for L2 Norm.
      model_wrapper.AddOps(
          ::qnn::BuildL2NormOp(model_wrapper.GetTensorPool(), input_tensors,
                               output_tensors, 9.99999997E-7),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflConv2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t dilation_w_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dDilationWOption(litert_op.Get(), &dilation_w_factor));
      int32_t dilation_h_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dDilationHOption(litert_op.Get(), &dilation_h_factor));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetConv2dFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildConv2dOp(model_wrapper.GetTensorPool(),
                                      input_tensors, {activation_input},
                                      stride_h, stride_w, dilation_h_factor,
                                      dilation_w_factor, qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflConv3d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_d;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dStrideDOption(litert_op.Get(), &stride_d));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dStrideHOption(litert_op.Get(), &stride_h));
      int32_t dilation_d_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dDilationDOption(litert_op.Get(), &dilation_d_factor));
      int32_t dilation_w_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dDilationWOption(litert_op.Get(), &dilation_w_factor));
      int32_t dilation_h_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv3dDilationHOption(litert_op.Get(), &dilation_h_factor));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetConv3dFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildConv3dOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input},
          stride_d, stride_h, stride_w, dilation_d_factor, dilation_h_factor,
          dilation_w_factor, qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTransposeConv: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTransposeConvPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTransposeConvStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTransposeConvStrideHOption(litert_op.Get(), &stride_h));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildTransposeConvOp(model_wrapper.GetTensorPool(),
                                             input_tensors, {activation_input},
                                             stride_h, stride_w, qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDepthwiseConv2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthwiseConv2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthwiseConv2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthwiseConv2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t dilation_w_factor;
      LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dDilationWOption(
          litert_op.Get(), &dilation_w_factor));
      int32_t dilation_h_factor;
      LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dDilationHOptions(
          litert_op.Get(), &dilation_h_factor));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildDepthwiseConv2dOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input},
          stride_h, stride_w, dilation_h_factor, dilation_w_factor,
          qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAveragePool2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetAveragePool2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetAveragePool2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetAveragePool2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t filter_width;
      LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFilterWidthOption(
          litert_op.Get(), &filter_width));
      int32_t filter_height;
      LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFilterHeightOption(
          litert_op.Get(), &filter_height));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildAveragePoolOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input},
          stride_h, stride_w, filter_height, filter_width, qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMaxPool2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMaxPool2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMaxPool2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMaxPool2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t filter_width;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMaxPool2dFilterWidthOption(litert_op.Get(), &filter_width));
      int32_t filter_height;
      LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dFilterHeightOption(
          litert_op.Get(), &filter_height));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildMaxPoolOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input},
          stride_h, stride_w, filter_height, filter_width, qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflL2Pool2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetL2Pool2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetL2Pool2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetL2Pool2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t filter_width;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetL2Pool2dFilterWidthOption(litert_op.Get(), &filter_width));
      int32_t filter_height;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetL2Pool2dFilterHeightOption(litert_op.Get(), &filter_height));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dFusedActivationOption(
          litert_op.Get(), &fused_activation));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));

      auto& activation_input = ::qnn::CreateFusedActivationInputTensor(
          model_wrapper.GetTensorPool(), fused_activation, output_tensors);
      auto ops = ::qnn::BuildL2PoolOp(
          model_wrapper.GetTensorPool(), input_tensors, {activation_input},
          stride_h, stride_w, filter_height, filter_width, qnn_padding);
      ::qnn::AddFusedActivationNode(ops, fused_activation, activation_input,
                                    output_tensors[0]);
      model_wrapper.AddOps(std::move(ops), prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDepthToSpace: {
      int32_t block_size;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthToSpaceBlockSizeOption(litert_op.Get(), &block_size));
      model_wrapper.AddOps(
          ::qnn::BuildDepthToSpaceOp(model_wrapper.GetTensorPool(),
                                     input_tensors, output_tensors, block_size),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSpaceToDepth: {
      int32_t block_size;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSpaceToDepthBlockSizeOption(litert_op.Get(), &block_size));
      model_wrapper.AddOps(
          ::qnn::BuildSpaceToDepthOp(model_wrapper.GetTensorPool(),
                                     input_tensors, output_tensors, block_size),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflHardSwish: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseHardSwishOp(model_wrapper.GetTensorPool(),
                                             input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLeakyRelu: {
      float alpha;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetLeakyReluAlphaOption(litert_op.Get(), &alpha));
      model_wrapper.AddOps(
          ::qnn::BuildLeakyReluOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, alpha),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflResizeBilinear: {
      bool align_corners;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearAlignCornersOption(
          litert_op.Get(), &align_corners));
      bool half_pixel_centers;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearHalfPixelCenterOption(
          litert_op.Get(), &half_pixel_centers));
      model_wrapper.AddOps(
          ::qnn::BuildResizeBilinearOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors,
                                       align_corners, half_pixel_centers),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflResizeNearestNeighbor: {
      bool align_corners;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeNearestNeighborAlignCornersOption(
          litert_op.Get(), &align_corners));
      bool half_pixel_centers;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetResizeNearestNeighborHalfPixelCenterOption(
              litert_op.Get(), &half_pixel_centers));
      model_wrapper.AddOps(
          ::qnn::BuildResizeNearestOp(model_wrapper.GetTensorPool(),
                                      input_tensors, output_tensors,
                                      align_corners, half_pixel_centers),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPad:
    case LiteRtOpCode::kLiteRtOpCodeTflPadv2: {
      model_wrapper.AddOps(
          ::qnn::BuildConstantPadOp(model_wrapper.GetTensorPool(),
                                    input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMirrorPad: {
      uint32_t mode;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMirrorPadModeOption(litert_op.Get(), &mode));
      model_wrapper.AddOps(
          ::qnn::BuildMirrorPadOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors, mode),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflCumsum: {
      bool exclusive;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetCumsumExclusiveOption(litert_op.Get(), &exclusive));
      bool reverse;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetCumsumReverseOption(litert_op.Get(), &reverse));
      model_wrapper.AddOps(
          ::qnn::BuildCumsumOp(model_wrapper.GetTensorPool(), input_tensors,
                               output_tensors, exclusive, reverse),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGatherNd: {
      model_wrapper.AddOps(
          ::qnn::BuildGatherNdOp(model_wrapper.GetTensorPool(), input_tensors,
                                 output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPow: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwisePowerOp(model_wrapper.GetTensorPool(),
                                         input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLessEqual: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseLessEqualOp(model_wrapper.GetTensorPool(),
                                             input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogicalNot: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseNotOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGreaterEqual: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseGreaterEqualOp(model_wrapper.GetTensorPool(),
                                                input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflExp: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseExpOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflEqual: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseEqualOp(model_wrapper.GetTensorPool(),
                                         input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLog: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseLogOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAbs: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseAbsOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReverseV2: {
      model_wrapper.AddOps(::qnn::BuildReverseOp(model_wrapper.GetTensorPool(),
                                                 input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflArgMax: {
      model_wrapper.AddOps(::qnn::BuildArgMaxOp(model_wrapper.GetTensorPool(),
                                                input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflArgMin: {
      model_wrapper.AddOps(::qnn::BuildArgMinOp(model_wrapper.GetTensorPool(),
                                                input_tensors, output_tensors),
                           prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflStridedSlice: {
      std::int32_t begin_mask;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetStridedSliceBeginMaskOption(litert_op.Get(), &begin_mask));
      std::int32_t end_mask;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetStridedSliceEndMaskOption(litert_op.Get(), &end_mask));
      std::int32_t ellipsis_mask;
      LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceEllipsisMaskOption(
          litert_op.Get(), &ellipsis_mask));
      std::int32_t shrink_axis_mask;
      LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceShrinkAxisMaskOption(
          litert_op.Get(), &shrink_axis_mask));
      std::int32_t new_axis_mask;
      LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceNewAxisMaskOption(
          litert_op.Get(), &new_axis_mask));
      bool offset;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetStridedSliceOffsetOption(litert_op.Get(), &offset));
      model_wrapper.AddOps(
          ::qnn::BuildStridedSliceOp(model_wrapper.GetTensorPool(),
                                     input_tensors, output_tensors, begin_mask,
                                     end_mask, ellipsis_mask, shrink_axis_mask,
                                     new_axis_mask, offset),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflNeg: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseNegOp(model_wrapper.GetTensorPool(),
                                       input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRound: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseRoundOp(model_wrapper.GetTensorPool(),
                                         input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSign: {
      model_wrapper.AddOps(
          ::qnn::BuildElementwiseSignOp(model_wrapper.GetTensorPool(),
                                        input_tensors, output_tensors),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogSoftmax: {
      std::uint32_t axis = input_tensors[0].get().GetRank() - 1;
      float beta{1.0};
      model_wrapper.AddOps(
          ::qnn::BuildLogSoftmaxOp(model_wrapper.GetTensorPool(), input_tensors,
                                   output_tensors, axis, beta),
          prefix, suffix);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflScatterNd: {
      model_wrapper.AddOps(
          ::qnn::BuildScatterNdOp(model_wrapper.GetTensorPool(), input_tensors,
                                  output_tensors),
          prefix, suffix);
      break;
    }
    default: {
      LITERT_LOG(LITERT_ERROR,
                 "LiteRT Op Code: %d is not supported in Qualcomm Compiler.",
                 litert_op.Code());
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus AddTensorToQnn(
    const QnnApi* qnn_api, Qnn_GraphHandle_t& graph_handle,
    ::qnn::TensorWrapper& tensor,
    absl::flat_hash_set<const ::qnn::TensorWrapper*>& created_tensors,
    bool use_qint16_as_quint16) {
  if (created_tensors.count(&tensor)) {
    return kLiteRtStatusOk;
  }

  if (use_qint16_as_quint16) {
    tensor.ConvertQint16ToQuint16();
  }

  auto error =
      qnn_api->tensorCreateGraphTensor(graph_handle, &(tensor.GetQnnTensor()));
  if (QNN_SUCCESS == error) {
    created_tensors.emplace(&tensor);
    return kLiteRtStatusOk;
  }

  const char* message = nullptr;
  auto get_message_error = qnn_api->errorGetMessage(error, &message);
  if (QNN_SUCCESS == get_message_error) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to create graph tensor, error: %d, message: %s", error,
               message);
  } else {
    LITERT_LOG(LITERT_ERROR,
               "Failed to create graph tensor and get error message, error: %d",
               error);
  }
  return kLiteRtStatusErrorRuntimeFailure;
}

LiteRtStatus MapGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                      Qnn_ProfileHandle_t profile_handle,
                      LiteRtSubgraph subgraph, absl::string_view qnn_graph_name,
                      const ::qnn::Options& options) {
  GraphMapper graph_mapper(subgraph, qnn, context_handle, profile_handle);
  LITERT_RETURN_IF_ERROR(graph_mapper.IsLiteRtSubgraphSupported());
  LITERT_RETURN_IF_ERROR(graph_mapper.InitQnnGraph(qnn_graph_name, options));

  //
  // Legalize subgraph inputs and update tensors in scope
  //

  ::qnn::ModelWrapper model_wrapper;
  absl::flat_hash_map<LiteRtTensor, ::qnn::TensorWrapper*>
      litert_tensor_to_wrapper;
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors;
  auto dump_ids = options.GetDumpTensorIds();
  absl::flat_hash_set<std::int32_t> ids_to_dump(dump_ids.begin(),
                                                dump_ids.end());

  for (const auto& subgraph_input : graph_mapper.Graph().Inputs()) {
    ::qnn::TensorWrapper* tensor_wrapper{nullptr};
    LITERT_RETURN_IF_ERROR(ConvertTensor(subgraph_input,
                                         model_wrapper.GetTensorPool(),
                                         tensor_wrapper, ids_to_dump));
    litert_tensor_to_wrapper.emplace(subgraph_input.Get(), tensor_wrapper);
    LITERT_RETURN_IF_ERROR(AddTensorToQnn(qnn.Api(), graph_mapper.QnnGraph(),
                                          *tensor_wrapper, created_tensors,
                                          options.GetUseQint16AsQuint16()));
  }

  for (const auto& subgraph_output : graph_mapper.Graph().Outputs()) {
    graph_mapper.RegisterOutput(subgraph_output.Get());
  }
  //
  // Topologically traverse graph, legalizing and updating tensors in scope
  //

  std::ostringstream dump;
  auto ops = graph_mapper.Graph().Ops();
  for (auto it = ops.begin(); it != ops.end(); ++it) {
    const auto& op = *it;
    std::vector<::qnn::TensorWrapperRef> input_tensors;
    for (const auto& input : op.Inputs()) {
      if (const auto it = litert_tensor_to_wrapper.find(input.Get());
          it == litert_tensor_to_wrapper.end()) {
        ::qnn::TensorWrapper* tensor_wrapper{nullptr};
        LITERT_RETURN_IF_ERROR(ConvertTensor(
            input, model_wrapper.GetTensorPool(), tensor_wrapper, ids_to_dump));
        // add into map to capture re-used static tensor
        litert_tensor_to_wrapper.emplace(input.Get(), tensor_wrapper);
        input_tensors.emplace_back(*tensor_wrapper);
      } else {
        input_tensors.emplace_back(*(it->second));
      }
    }

    std::vector<::qnn::TensorWrapperRef> output_tensors;
    for (const auto& output : op.Outputs()) {
      bool is_tensor_read_and_write = graph_mapper.IsTensorOutput(output.Get());
      ::qnn::TensorWrapper* tensor_wrapper{nullptr};
      LITERT_RETURN_IF_ERROR(
          ConvertTensor(output, model_wrapper.GetTensorPool(), tensor_wrapper,
                        ids_to_dump, is_tensor_read_and_write));
      litert_tensor_to_wrapper.emplace(output.Get(), tensor_wrapper);
      output_tensors.emplace_back(*tensor_wrapper);
    }

    // Add litert op id to qnn op name to preserve op mapping.
    std::string op_namespace = "";
    std::string op_index = absl::StrCat(
        "_LiteRt_OpId_", std::to_string(std::distance(std::begin(ops), it)));

    if (!op.Outputs().empty()) {
      // Add op namespace inference based on output tensor names.
      std::vector<std::string> candidate_names;
      for (const auto& output_tensor : op.Outputs()) {
        for (const auto& name :
             absl::StrSplit(output_tensor.Name(), ';', absl::SkipEmpty())) {
          candidate_names.emplace_back(name);
        }
      }
      op_namespace =
          absl::StrCat(TfliteNodeNamespaceHeuristic(GetTfliteOpName(op.Code()),
                                                    candidate_names),
                       "/");
    }
    LITERT_RETURN_IF_ERROR(ConvertOp(options.GetUseHtpPreference(), op,
                                     model_wrapper, input_tensors,
                                     output_tensors, op_namespace, op_index));
  }
  // TODO (jiunkaiy): Set this graph-to-graph transformation as a compile flag.
  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOptPrefill;
  GraphToGraphTransform(g2g_option, model_wrapper,
                        [api = qnn.Api(), backend = qnn.BackendHandle()](
                            ::qnn::OpWrapper& op) -> bool {
                          return QNN_SUCCESS == api->backendValidateOpConfig(
                                                    backend, op.GetOpConfig());
                        });

  // Create ops and their corresponding tensors.
  for (auto& op_wrapper : model_wrapper.GetOps()) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      LITERT_RETURN_IF_ERROR(AddTensorToQnn(
          qnn.Api(), graph_mapper.QnnGraph(), tensor_wrapper_ref.get(),
          created_tensors, options.GetUseQint16AsQuint16()));
    }
    auto error = qnn.Api()->graphAddNode(graph_mapper.QnnGraph(),
                                         op_wrapper.GetOpConfig());
    if (QNN_SUCCESS == error) {
      continue;
    }

    const char* message = nullptr;
    auto get_message_error = qnn.Api()->errorGetMessage(error, &message);
    if (QNN_SUCCESS == get_message_error) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to add node into graph, error: %d, message: %s", error,
                 message);
    } else {
      LITERT_LOG(
          LITERT_ERROR,
          "Failed to add node into graph and get error message, error: %d",
          error);
    }
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Dump IR Json to understand Qnn graph.
  if (!options.GetIrJsonDir().empty()) {
    ::qnn::DumpIrJson(created_tensors, model_wrapper.GetOps(),
                      options.GetIrJsonDir(), qnn_graph_name);
  }

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Finalize());

  return kLiteRtStatusOk;
}

//===----------------------------------------------------------------------===//
//
//                                           [WIP] LiteRT SUBGRAPH -> QNN GRAPH
//
// Core driver for IR translation. Traverses LiteRt Subgraph, iteratively
// "legalizing" (mapping) LiteRt entities to their QNN counterpart.
//
// APPROACH:
//
// To support the general case we will need a driver loop that either
// traverses input recursively through edges or just iterates topologically.
//
// The algorithm is pretty straightforward:
// * Store mapping between already evaluated LiteRtTensors and their
//   newly constructed Qnn Tensor counterpart.
// * Look up QNN Tensors when setting QNN Op inputs.
// * Add new QNN Tensor when setting QNN Op outputs.
//
// NOTES ON QNN API:
//
// After QNN Tensors are registered in the context, they need only
// be stored as their ID. QNN Tensor and "id" : uint32_t are used
// interchangeably.
//
//===----------------------------------------------------------------------===//

LiteRtStatus ComposeGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                          Qnn_ProfileHandle_t profile_handle,
                          LiteRtSubgraph subgraph,
                          absl::string_view qnn_graph_name,
                          const ::qnn::Options& options) {
  LITERT_RETURN_IF_ERROR(MapGraph(qnn, context_handle, profile_handle, subgraph,
                                  qnn_graph_name, options));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
