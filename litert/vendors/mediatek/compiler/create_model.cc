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

#include "litert/vendors/mediatek/compiler/create_model.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "neuron/api/NeuronAdapter.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/mediatek/compiler/legalizations/add_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/batch_matmul_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/common_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/concat_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/fully_connected_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/gelu_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/legalize_helper.h"
#include "litert/vendors/mediatek/compiler/legalizations/mean_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/mul_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/neuron_utils.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/compiler/legalizations/reshape_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/resize_bilinear_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/resize_nearest_neighbor_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/rms_norm_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/rsqrt_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/softmax_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/split_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/squared_difference_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/strided_slice_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/sub_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/transpose_conv_op_legalization.h"
#include "litert/vendors/mediatek/compiler/legalizations/transpose_op_legalization.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<void> CreateModel(const NeuronAdapterApi& neuron_adapter_api,
                           const litert::Subgraph& partition,
                           const std::string& model_name, NeuronModel* model,
                           OperandMap* operand_map,
                           std::unordered_set<int>* unknown_op_indices) {
  if (neuron_adapter_api.api().model_set_name(model, model_name.c_str()) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to set model name");
  }

  std::vector<uint32_t> input_indices;
  int32_t tensor_flags =
      unknown_op_indices ? NN_TENSOR_FLAG_USE_INVALID_TENSOR_TYPE : 0;
  for (const auto& input : partition.Inputs()) {
    auto operand_index = operand_map->GetOperandIndex(input, tensor_flags);
    if (!operand_index) {
      return operand_index.Error();
    }
    input_indices.push_back(*operand_index);
  }

  std::vector<uint32_t> output_indices;
  for (const auto& output : partition.Outputs()) {
    auto operand_index = operand_map->GetOperandIndex(output, tensor_flags);
    if (!operand_index) {
      return operand_index.Error();
    }
    output_indices.push_back(*operand_index);
  }

  if (neuron_adapter_api.api().model_identify_inputs_and_outputs(
          model, input_indices.size(), input_indices.data(),
          output_indices.size(), output_indices.data()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to identify model I/Os");
  }

  const auto& ops = partition.Ops();
  for (int op_idx = 0; op_idx < ops.size(); ++op_idx) {
    LITERT_LOG(LITERT_INFO, "Legalizing op index: %d", op_idx);
    const auto& op = ops[op_idx];
    Expected<void> status;
    // Map un-legalized op as unknown op to by-pass the legalization
    if (unknown_op_indices && unknown_op_indices->count(op_idx)) {
      LITERT_RETURN_IF_ERROR(LegalizeCommonOp(
          neuron_adapter_api, model, *operand_map, op, NEURON_UNKNOWN));
      continue;
    }
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
        status = LegalizeAddOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflMul:
        status = LegalizeMulOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflBatchMatmul:
        status =
            LegalizeBatchMatMulOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflFullyConnected:
        status = LegalizeFullyConnectedOp(neuron_adapter_api, model,
                                          *operand_map, op);
        break;
      case kLiteRtOpCodeTflReshape:
        status = LegalizeReshapeOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflTranspose:
        status =
            LegalizeTransposeOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflRsqrt:
        status = LegalizeRsqrtOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflConcatenation:
        status = LegalizeConcatOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflQuantize:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_QUANTIZE);
        break;
      case kLiteRtOpCodeTflSlice:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_SLICE);
        break;
      case kLiteRtOpCodeTflStridedSlice:
        status =
            LegalizeStridedSliceOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflSplit:
        status = LegalizeSplitOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflTanh:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_TANH);
        break;
      case kLiteRtOpCodeTflSub:
        status = LegalizeSubOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflSoftmax:
        status = LegalizeSoftmaxOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflMean:
        status = LegalizeMeanOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflGelu:
        status = LegalizeGeluOp(neuron_adapter_api, model, *operand_map, op);
        break;
      case kLiteRtOpCodeTflMaxPool2d:
        status = LegalizeOp(
            neuron_adapter_api, model, *operand_map, op, NEURON_MAX_POOL_2D,
            std::make_tuple(AddMaxPool2dPaddingOption,           // padding
                            AddMaxPool2dStrideWOption,           // stride_w
                            AddMaxPool2dStrideHOption,           // stride_h
                            AddMaxPool2dFilterWOption,           // filter_w
                            AddMaxPool2dFilterHOption,           // filter_h
                            AddMaxPool2dFuseActivationOption));  // activation
        break;
      case kLiteRtOpCodeTflHardSwish:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_HARD_SWISH);
        break;
      case kLiteRtOpCodeTflAveragePool2d:
        status = LegalizeOp(
            neuron_adapter_api, model, *operand_map, op, NEURON_AVERAGE_POOL_2D,
            std::make_tuple(
                AddAveragePool2dPaddingOption,           // padding
                AddAveragePool2dStrideWOption,           // stride_w
                AddAveragePool2dStrideHOption,           // stride_h
                AddAveragePool2dFilterWOption,           // filter_w
                AddAveragePool2dFilterHOption,           // filter_h
                AddAveragePool2dFuseActivationOption));  // activation
        break;
      case kLiteRtOpCodeTflPad:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_PAD);
        break;
      case kLiteRtOpCodeTflLogistic:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_LOGISTIC);
        break;
      case kLiteRtOpCodeTflSum:
        status = LegalizeOp(neuron_adapter_api, model, *operand_map, op,
                            NEURON_REDUCE_SUM,
                            std::make_tuple(AddSumKeepDimsOption));
        break;
      case kLiteRtOpCodeTflConv2d:
        status = LegalizeOp(
            neuron_adapter_api, model, *operand_map, op, NEURON_CONV_2D,
            std::make_tuple(AddConv2dPaddingOption,         // padding
                            AddConv2dStrideWOption,         // stride_w
                            AddConv2dStrideHOption,         // stride_h
                            AddConv2dFuseActivationOption,  // activation
                            AddConv2dDataOption,            // data format
                            AddConv2dDilationWOption,       // dilation_w
                            AddConv2dDilationHOption));     // dilation_h
        break;
      case kLiteRtOpCodeTflDepthwiseConv2d:
        status = LegalizeOp(
            neuron_adapter_api, model, *operand_map, op,
            NEURON_DEPTHWISE_CONV_2D,
            std::make_tuple(
                AddDepthwiseConv2dPaddingOption,  // padding
                AddDepthwiseConv2dStrideWOption,  // stride_w
                AddDepthwiseConv2dStrideHOption,  // stride_h
                AddDepthwiseConv2dDepthMultiplierOption,
                AddDepthwiseConv2dFuseActivationOption,  // activation
                AddDepthwiseConv2dDataOption,            // data format
                AddDepthwiseConv2dDilationWOption,       // dilation_w
                AddDepthwiseConv2dDilationHOption));     // dilation_h
        break;
      case kLiteRtOpCodeTflSquaredDifference:
        status = LegalizeSquaredDifferenceOp(neuron_adapter_api, model,
                                             *operand_map, op);
        break;
      case kLiteRtOpCodeTflResizeBilinear:
        status = LegalizeResizeBilinearOp(neuron_adapter_api, model,
                                          *operand_map, op);
        break;
      case kLiteRtOpCodeTflResizeNearestNeighbor:
        status = LegalizeResizeNearestNeighborOp(neuron_adapter_api, model,
                                                 *operand_map, op);
        break;
      case kLiteRtOpCodeTflTransposeConv:
        status = LegalizeTransposeConvOp(neuron_adapter_api, model,
                                         *operand_map, op);
        break;
      case kLiteRtOpCodeTflDequantize:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_DEQUANTIZE);
        break;
      case kLiteRtOpCodeTflPadv2:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_PAD_V2);
        break;
      case kLiteRtOpCodeTflReduceMax:
        status = LegalizeOp(neuron_adapter_api, model, *operand_map, op,
                            NEURON_REDUCE_MAX,
                            std::make_tuple(AddReduceMaxKeepDimsOption));
        break;
      case kLiteRtOpCodeTflSqrt:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_SQRT);
        break;
      case kLiteRtOpCodeTflDiv:
        status =
            LegalizeOp(neuron_adapter_api, model, *operand_map, op, NEURON_DIV,
                       std::make_tuple(AddDivFuseActivationOption));
        break;
      case kLiteRtOpCodeTflCast:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_CAST);
        break;
      case kLiteRtOpCodeTflPrelu:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_PRELU);
        break;
      case kLiteRtOpCodeTflMaximum:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_MAXIMUM);
        break;
      case kLiteRtOpCodeTflRelu:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_RELU);
        break;
      case kLiteRtOpCodeTflAbs:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_ABS);
        break;
      case kLiteRtOpCodeTflGreater:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_GREATER);
        break;
      case kLiteRtOpCodeTflMinimum:
        status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                  NEURON_MINIMUM);
        break;
      case kLiteRtOpCodeShloComposite:
        const char* op_name;
        if (LiteRtGetSHLOCompositeOpName(op.Get(), &op_name) !=
            kLiteRtStatusOk) {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                       "LiteRtGetSHLOCompositeOpName returns an error");
        }
        if (std::string(op_name) == "odml.rms_norm") {
          status =
              LegalizeRmsNormOp(neuron_adapter_api, model, *operand_map, op);
        } else if (std::string(op_name) == "odml.l2_norm") {
          status = LegalizeCommonOp(neuron_adapter_api, model, *operand_map, op,
                                    NEURON_L2_NORMALIZATION);
        } else {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                       "Unsupported ShloComposite op");
        }
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unsupported op: %d", op.Code());
        return Error(kLiteRtStatusErrorRuntimeFailure, "Unsupported op");
    }

    if (!status) {
      return status.Error();
    }
  }

  if (neuron_adapter_api.api().model_finish(model) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to finish model");
  }

  return {};
}

}  // namespace litert::mediatek
