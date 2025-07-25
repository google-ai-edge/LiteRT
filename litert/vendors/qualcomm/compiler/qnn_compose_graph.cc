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
#include <cstdint>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_op_options.h"
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
#include "litert/vendors/qualcomm/core/builders/hard_swish_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/leaky_relu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/logistic_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/mean_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pad_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pool2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reduce_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu6_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/resize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reverse_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/rms_norm_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/spatial_transform_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/strided_slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/tanh_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_conv_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
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
            absl::Span<const float>{per_channel_quant.scales,
                                    per_channel_quant.num_channels},
            absl::Span<const std::int32_t>{zero_points.data(),
                                           zero_points.size()});
      } else {
        quantize_params.emplace<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
            per_channel_quant.quantized_dimension,
            absl::Span<const float>{per_channel_quant.scales,
                                    per_channel_quant.num_channels},
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
        reinterpret_cast<const void*>(litert_tensor.Weights().Bytes().data()));
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
                       ::qnn::TensorPool& tensor_pool,
                       std::vector<::qnn::TensorWrapperRef>& input_tensors,
                       std::vector<::qnn::TensorWrapperRef>& output_tensors,
                       std::vector<::qnn::OpWrapper>& op_wrappers) {
  switch (litert_op.Code()) {
    case LiteRtOpCode::kLiteRtOpCodeTflCast: {
      op_wrappers =
          ::qnn::BuildCastOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflConcatenation: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConcatenationAxisOption(litert_op.Get(), &axis));
      uint32_t fused_activation;
      LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildConcatenationOp(tensor_pool, input_tensors,
                                                output_tensors, axis);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAdd: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetAddFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildElementwiseAddOp(tensor_pool, input_tensors,
                                                 output_tensors);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogicalAnd: {
      op_wrappers = ::qnn::BuildElementwiseAndOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflBroadcastTo: {
      op_wrappers =
          ::qnn::BuildBroadcastToOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflCos: {
      op_wrappers = ::qnn::BuildElementwiseCosOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDiv: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetDivFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildElementwiseDivOp(tensor_pool, input_tensors,
                                                 output_tensors);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGreater: {
      op_wrappers = ::qnn::BuildElementwiseGreaterOp(tensor_pool, input_tensors,
                                                     output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLess: {
      op_wrappers = ::qnn::BuildElementwiseLessOp(tensor_pool, input_tensors,
                                                  output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMul: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetMulFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildElementwiseMulOp(tensor_pool, input_tensors,
                                                 output_tensors);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRsqrt: {
      op_wrappers = ::qnn::BuildElementwiseRsqrtOp(tensor_pool, input_tensors,
                                                   output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSqrt: {
      op_wrappers = ::qnn::BuildElementwiseSqrtOp(tensor_pool, input_tensors,
                                                  output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSin: {
      op_wrappers = ::qnn::BuildElementwiseSinOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSquaredDifference: {
      op_wrappers = ::qnn::BuildElementwiseSquaredDifferenceOp(
          tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSquare: {
      op_wrappers = ::qnn::BuildElementwiseSquareOp(tensor_pool, input_tensors,
                                                    output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSub: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetSubFusedActivationOption(
          litert_op.Get(), &fused_activation));

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildElementwiseSubOp(tensor_pool, input_tensors,
                                                 output_tensors);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMinimum: {
      op_wrappers = ::qnn::BuildElementwiseMinimumOp(tensor_pool, input_tensors,
                                                     output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMaximum: {
      op_wrappers = ::qnn::BuildElementwiseMaximumOp(tensor_pool, input_tensors,
                                                     output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflFloorDiv: {
      op_wrappers = ::qnn::BuildElementwiseFloorDivOp(
          tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflNotEqual: {
      op_wrappers = ::qnn::BuildElementwiseNotEqualOp(
          tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflEmbeddingLookup: {
      op_wrappers = ::qnn::BuildEmbeddingLookupOp(tensor_pool, input_tensors,
                                                  output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflFullyConnected: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedFusedActivationOption(
          litert_op.Get(), &fused_activation));
      bool keep_num_dims{};
      LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedKeepNumDimsOption(
          litert_op.Get(), &keep_num_dims));

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      if (use_htp_preferences) {
        op_wrappers = ::qnn::BuildFullyConnectedOpHtp(
            tensor_pool, input_tensors, output_tensors, keep_num_dims);
      }
      if (op_wrappers.empty()) {
        op_wrappers = ::qnn::BuildFullyConnectedOp(
            tensor_pool, input_tensors, output_tensors, keep_num_dims);
      }
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGather: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(litert_op.Get(), &axis));
      int32_t batch_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetGatherBatchDimsOption(litert_op.Get(), &batch_dims));
      op_wrappers = ::qnn::BuildGatherOp(tensor_pool, input_tensors,
                                         output_tensors, axis, batch_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGelu: {
      op_wrappers =
          ::qnn::BuildGeluOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRelu: {
      op_wrappers =
          ::qnn::BuildReluOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRelu6: {
      op_wrappers =
          ::qnn::BuildRelu6Op(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogistic: {
      op_wrappers =
          ::qnn::BuildLogisticOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflBatchMatmul: {
      bool adj_x{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetBatchMatmulAdjXOption(litert_op.Get(), &adj_x));
      bool adj_y{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetBatchMatmulAdjYOption(litert_op.Get(), &adj_y));
      op_wrappers = ::qnn::BuildMatmulOp(tensor_pool, input_tensors,
                                         output_tensors, adj_x, adj_y);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMean: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMeanKeepDimsOption(litert_op.Get(), &keep_dims));
      op_wrappers = ::qnn::BuildMeanOp(tensor_pool, input_tensors,
                                       output_tensors, keep_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflQuantize: {
      op_wrappers =
          ::qnn::BuildQuantizeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDequantize: {
      op_wrappers =
          ::qnn::BuildDequantizeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSum: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSumKeepDimsOption(litert_op.Get(), &keep_dims));
      op_wrappers = ::qnn::BuildReduceSumOp(tensor_pool, input_tensors,
                                            output_tensors, keep_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReduceMax: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetReduceMaxKeepDimsOption(litert_op.Get(), &keep_dims));
      op_wrappers = ::qnn::BuildReduceMaxOp(tensor_pool, input_tensors,
                                            output_tensors, keep_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReshape: {
      op_wrappers =
          ::qnn::BuildReshapeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSelect:
    case LiteRtOpCode::kLiteRtOpCodeTflSelectV2: {
      op_wrappers =
          ::qnn::BuildSelectOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSlice: {
      op_wrappers =
          ::qnn::BuildSliceOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSoftmax: {
      float beta{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSoftmaxBetaOption(litert_op.Get(), &beta));
      op_wrappers = ::qnn::BuildSoftmaxOp(tensor_pool, input_tensors,
                                          output_tensors, beta);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSplit: {
      int32_t num_splits{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSplitNumSplitsOption(litert_op.Get(), &num_splits));
      op_wrappers = ::qnn::BuildSplitOp(tensor_pool, input_tensors,
                                        output_tensors, num_splits);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTanh: {
      op_wrappers =
          ::qnn::BuildTanhOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTranspose: {
      op_wrappers =
          ::qnn::BuildTransposeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPack: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(litert_op.Get(), &axis));
      op_wrappers =
          ::qnn::BuildPackOp(tensor_pool, input_tensors, output_tensors, axis);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflUnpack: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetUnpackAxisOption(litert_op.Get(), &axis));
      op_wrappers = ::qnn::BuildUnpackOp(tensor_pool, input_tensors,
                                         output_tensors, axis);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDynamicUpdateSlice: {
      op_wrappers = ::qnn::BuildDynamicUpdateSliceOp(tensor_pool, input_tensors,
                                                     output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeShloComposite: {
      auto info = GetOptionsAs<CompositeOptions>(litert_op.Get());
      if (!info) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (info->name == CompositeOptions::kRmsNorm) {
        // TODO(yunandrew): Support custom epsilon for RMS Norm.
        float epsilon = 9.99999997E-7;
        op_wrappers = ::qnn::BuildRmsNormOp(tensor_pool, input_tensors,
                                            output_tensors, epsilon);
      }
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

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildConv2dOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          dilation_h_factor, dilation_w_factor, qnn_padding);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
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

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildConv3dOp(
          tensor_pool, input_tensors, output_tensors, stride_d, stride_h,
          stride_w, dilation_d_factor, dilation_h_factor, dilation_w_factor,
          qnn_padding);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
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

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildTransposeConvOp(tensor_pool, input_tensors,
                                                output_tensors, stride_h,
                                                stride_w, qnn_padding);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
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

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildDepthwiseConv2dOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          dilation_h_factor, dilation_w_factor, qnn_padding);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
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

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildAveragePoolOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          filter_height, filter_width, qnn_padding);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
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

      auto& activation_output = ::qnn::ReplaceOutputTensorForFusedActivation(
          tensor_pool, fused_activation, output_tensors);
      op_wrappers = ::qnn::BuildMaxPoolOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          filter_height, filter_width, qnn_padding);
      ::qnn::AddFusedActivationNode(op_wrappers, fused_activation,
                                    output_tensors[0], activation_output);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDepthToSpace: {
      int32_t block_size;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthToSpaceBlockSizeOption(litert_op.Get(), &block_size));
      op_wrappers = ::qnn::BuildDepthToSpaceOp(tensor_pool, input_tensors,
                                               output_tensors, block_size);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSpaceToDepth: {
      int32_t block_size;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSpaceToDepthBlockSizeOption(litert_op.Get(), &block_size));
      op_wrappers = ::qnn::BuildSpaceToDepthOp(tensor_pool, input_tensors,
                                               output_tensors, block_size);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflHardSwish: {
      op_wrappers =
          ::qnn::BuildHardSwishOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLeakyRelu: {
      float alpha;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetLeakyReluAlphaOption(litert_op.Get(), &alpha));
      op_wrappers = ::qnn::BuildLeakyReluOp(tensor_pool, input_tensors,
                                            output_tensors, alpha);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflResizeBilinear: {
      bool align_corners;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearAlignCornersOption(
          litert_op.Get(), &align_corners));
      bool half_pixel_centers;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearHalfPixelCenterOption(
          litert_op.Get(), &half_pixel_centers));
      op_wrappers = ::qnn::BuildResizeBilinearOp(tensor_pool, input_tensors,
                                                 output_tensors, align_corners,
                                                 half_pixel_centers);
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
      op_wrappers = ::qnn::BuildResizeNearestOp(tensor_pool, input_tensors,
                                                output_tensors, align_corners,
                                                half_pixel_centers);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPad:
    case LiteRtOpCode::kLiteRtOpCodeTflPadv2: {
      op_wrappers =
          ::qnn::BuildPadOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflCumsum: {
      bool exclusive;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetCumsumExclusiveOption(litert_op.Get(), &exclusive));
      bool reverse;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetCumsumReverseOption(litert_op.Get(), &reverse));
      op_wrappers = ::qnn::BuildCumsumOp(tensor_pool, input_tensors,
                                         output_tensors, exclusive, reverse);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGatherNd: {
      op_wrappers =
          ::qnn::BuildGatherNdOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPow: {
      op_wrappers = ::qnn::BuildElementwisePowerOp(tensor_pool, input_tensors,
                                                   output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLessEqual: {
      op_wrappers = ::qnn::BuildElementwiseLessEqualOp(
          tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogicalNot: {
      op_wrappers = ::qnn::BuildElementwiseNotOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGreaterEqual: {
      op_wrappers = ::qnn::BuildElementwiseGreaterEqualOp(
          tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflExp: {
      op_wrappers = ::qnn::BuildElementwiseExpOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflEqual: {
      op_wrappers = ::qnn::BuildElementwiseEqualOp(tensor_pool, input_tensors,
                                                   output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLog: {
      op_wrappers = ::qnn::BuildElementwiseLogOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAbs: {
      op_wrappers = ::qnn::BuildElementwiseAbsOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReverseV2: {
      op_wrappers =
          ::qnn::BuildReverseOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflArgMax: {
      op_wrappers =
          ::qnn::BuildArgMaxOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflArgMin: {
      op_wrappers =
          ::qnn::BuildArgMinOp(tensor_pool, input_tensors, output_tensors);
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
      op_wrappers = ::qnn::BuildStridedSliceOp(
          tensor_pool, input_tensors, output_tensors, begin_mask, end_mask,
          ellipsis_mask, shrink_axis_mask, new_axis_mask, offset);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflNeg: {
      op_wrappers = ::qnn::BuildElementwiseNegOp(tensor_pool, input_tensors,
                                                 output_tensors);
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

void AddTensorToQnn(
    const QnnApi* qnn_api, Qnn_GraphHandle_t& graph_handle,
    ::qnn::TensorWrapper& tensor,
    absl::flat_hash_set<const ::qnn::TensorWrapper*>& created_tensors,
    bool use_qint16_as_quint16) {
  if (!created_tensors.count(&tensor)) {
    if (use_qint16_as_quint16) {
      tensor.ConvertQint16ToQuint16();
    }
    qnn_api->tensorCreateGraphTensor(graph_handle, &(tensor.GetQnnTensor()));
    created_tensors.emplace(&tensor);
  }
}

LiteRtStatus MapGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                      LiteRtSubgraph subgraph, absl::string_view qnn_graph_name,
                      const ::qnn::Options& options) {
  GraphMapper graph_mapper(subgraph, qnn, context_handle);
  LITERT_RETURN_IF_ERROR(graph_mapper.IsLiteRtSubgraphSupported());
  LITERT_RETURN_IF_ERROR(graph_mapper.InitQnnGraph(qnn_graph_name));

  //
  // Legalize subgraph inputs and update tensors in scope
  //

  ::qnn::TensorPool tensor_pool;
  absl::flat_hash_map<LiteRtTensor, ::qnn::TensorWrapper*>
      litert_tensor_to_wrapper;
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors;
  auto dump_ids = options.GetDumpTensorIds();
  absl::flat_hash_set<std::int32_t> ids_to_dump(dump_ids.begin(),
                                                dump_ids.end());

  for (const auto& subgraph_input : graph_mapper.Graph().Inputs()) {
    ::qnn::TensorWrapper* tensor_wrapper{nullptr};
    LITERT_RETURN_IF_ERROR(ConvertTensor(subgraph_input, tensor_pool,
                                         tensor_wrapper, ids_to_dump));
    litert_tensor_to_wrapper.emplace(subgraph_input.Get(), tensor_wrapper);
    AddTensorToQnn(qnn.Api(), graph_mapper.QnnGraph(), *tensor_wrapper,
                   created_tensors, options.GetUseQint16AsQuint16());
  }

  for (const auto& subgraph_output : graph_mapper.Graph().Outputs()) {
    graph_mapper.RegisterOutput(subgraph_output.Get());
  }
  //
  // Topologically traverse graph, legalizing and updating tensors in scope
  //

  // TODO: make ConvertOp accept a vector and append OpWrapper in it.
  std::vector<::qnn::OpWrapper> graph_op_wrappers;
  std::ostringstream dump;
  for (const auto& op : graph_mapper.Graph().Ops()) {
    std::vector<::qnn::TensorWrapperRef> input_tensors;
    for (const auto& input : op.Inputs()) {
      if (const auto it = litert_tensor_to_wrapper.find(input.Get());
          it == litert_tensor_to_wrapper.end()) {
        ::qnn::TensorWrapper* tensor_wrapper{nullptr};
        LITERT_RETURN_IF_ERROR(
            ConvertTensor(input, tensor_pool, tensor_wrapper, ids_to_dump));
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
      LITERT_RETURN_IF_ERROR(ConvertTensor(output, tensor_pool, tensor_wrapper,
                                           ids_to_dump,
                                           is_tensor_read_and_write));
      litert_tensor_to_wrapper.emplace(output.Get(), tensor_wrapper);
      output_tensors.emplace_back(*tensor_wrapper);
    }

    std::vector<::qnn::OpWrapper> op_wrappers;
    LITERT_RETURN_IF_ERROR(ConvertOp(options.GetUseHtpPreference(), op,
                                     tensor_pool, input_tensors, output_tensors,
                                     op_wrappers));
    std::move(op_wrappers.begin(), op_wrappers.end(),
              std::back_inserter(graph_op_wrappers));
  }
  // TODO (jiunkaiy): Set this graph-to-graph transformation as a compile flag.
  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOptPrefill;
  GraphToGraphTransform(g2g_option, graph_op_wrappers, tensor_pool,
                        [api = qnn.Api(), backend = qnn.BackendHandle()](
                            ::qnn::OpWrapper& op) -> bool {
                          return QNN_SUCCESS == api->backendValidateOpConfig(
                                                    backend, op.GetOpConfig());
                        });

  // Create ops and their corresponding tensors.
  for (auto& op_wrapper : graph_op_wrappers) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      AddTensorToQnn(qnn.Api(), graph_mapper.QnnGraph(),
                     tensor_wrapper_ref.get(), created_tensors,
                     options.GetUseQint16AsQuint16());
    }
    qnn.Api()->graphAddNode(graph_mapper.QnnGraph(), op_wrapper.GetOpConfig());
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
                          LiteRtSubgraph subgraph,
                          absl::string_view qnn_graph_name,
                          const ::qnn::Options& options) {
  LITERT_RETURN_IF_ERROR(
      MapGraph(qnn, context_handle, subgraph, qnn_graph_name, options));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
