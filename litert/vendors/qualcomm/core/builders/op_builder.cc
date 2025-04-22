// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/op_builder.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::pair<std::uint32_t, std::uint32_t> ComputePaddingBeforeAfter(
    const std::uint32_t input_size, const std::uint32_t filter_size,
    const std::uint32_t stride, const std::uint32_t dilation_rate,
    const PaddingType padding_type) {
  // padding_before, padding_after
  std::pair<std::uint32_t, std::uint32_t> result{0, 0};
  if (stride == 0) {
    QNN_LOG_ERROR("Stride is 0");
    return result;
  }

  std::uint32_t output_size{};
  std::uint32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;

  switch (padding_type) {
    case PaddingType::Same:
      output_size = (input_size + stride - 1) / stride;
      break;
    case PaddingType::Valid:
      output_size = (input_size + stride - effective_filter_size) / stride;
      break;
    default:  // PaddingType::Unknown
      QNN_LOG_ERROR("Unknown padding type");
      return result;
  }

  std::int32_t total_padding =
      (output_size - 1) * stride + effective_filter_size - input_size;
  total_padding = total_padding > 0 ? total_padding : 0;
  result.first = total_padding / 2;
  result.second = result.first + total_padding % 2;
  return result;
}

OpWrapper& CreateOpWrapper(std::vector<OpWrapper>& ops, const char* op_type) {
  // TODO(jiunkaiy): Pass QnnOpCode in each opbuilder.
  static std::unordered_map<std::string_view, QnnOpCode> code_type_map = {
      {QNN_OP_ARGB_TO_RGB, QnnOpCode::kArgbToRgb},
      {QNN_OP_ARGMAX, QnnOpCode::kArgmax},
      {QNN_OP_ARGMIN, QnnOpCode::kArgmin},
      {QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM,
       QnnOpCode::kAxisAlignedBboxTransform},
      {QNN_OP_BATCHNORM, QnnOpCode::kBatchnorm},
      {QNN_OP_BATCH_PERMUTATION, QnnOpCode::kBatchPermutation},
      {QNN_OP_BATCH_TO_SPACE, QnnOpCode::kBatchToSpace},
      {QNN_OP_BBOX_TRANSFORM, QnnOpCode::kBboxTransform},
      {QNN_OP_BOX_WITH_NMS_LIMIT, QnnOpCode::kBoxWithNmsLimit},
      {QNN_OP_BUFFER, QnnOpCode::kBuffer},
      {QNN_OP_CAST, QnnOpCode::kCast},
      {QNN_OP_CHANNEL_SHUFFLE, QnnOpCode::kChannelShuffle},
      {QNN_OP_COL2_IM, QnnOpCode::kCol2Im},
      {QNN_OP_COLLECT_RPN_PROPOSALS, QnnOpCode::kCollectRpnProposals},
      {QNN_OP_COMBINED_NMS, QnnOpCode::kCombinedNms},
      {QNN_OP_CONCAT, QnnOpCode::kConcat},
      {QNN_OP_CONSTANT_OF_SHAPE, QnnOpCode::kConstantOfShape},
      {QNN_OP_CONV_1D, QnnOpCode::kConv1d},
      {QNN_OP_CONV_2D, QnnOpCode::kConv2d},
      {QNN_OP_CONV_3D, QnnOpCode::kConv3d},
      {QNN_OP_CONVERT, QnnOpCode::kConvert},
      {QNN_OP_CORRELATION_1D, QnnOpCode::kCorrelation1D},
      {QNN_OP_CREATE_SPARSE, QnnOpCode::kCreateSparse},
      {QNN_OP_CROP_AND_RESIZE, QnnOpCode::kCropAndResize},
      {QNN_OP_CUMULATIVE_SUM, QnnOpCode::kCumulativeSum},
      {QNN_OP_DEPTH_TO_SPACE, QnnOpCode::kDepthToSpace},
      {QNN_OP_DEPTH_WISE_CONV_1D, QnnOpCode::kDepthWiseConv1d},
      {QNN_OP_DEPTH_WISE_CONV_2D, QnnOpCode::kDepthWiseConv2d},
      {QNN_OP_DEQUANTIZE, QnnOpCode::kDequantize},
      {QNN_OP_DETECTION_OUTPUT, QnnOpCode::kDetectionOutput},
      {QNN_OP_DISTRIBUTE_FPN_PROPOSALS, QnnOpCode::kDistributeFpnProposals},
      {QNN_OP_ELEMENT_WISE_ABS, QnnOpCode::kElementWiseAbs},
      {QNN_OP_ELEMENT_WISE_ADD, QnnOpCode::kElementWiseAdd},
      {QNN_OP_ELEMENT_WISE_AND, QnnOpCode::kElementWiseAnd},
      {QNN_OP_ELEMENT_WISE_ASIN, QnnOpCode::kElementWiseAsin},
      {QNN_OP_ELEMENT_WISE_ATAN, QnnOpCode::kElementWiseAtan},
      {QNN_OP_ELEMENT_WISE_BINARY, QnnOpCode::kElementWiseBinary},
      {QNN_OP_ELEMENT_WISE_CEIL, QnnOpCode::kElementWiseCeil},
      {QNN_OP_ELEMENT_WISE_COS, QnnOpCode::kElementWiseCos},
      {QNN_OP_ELEMENT_WISE_DIVIDE, QnnOpCode::kElementWiseDivide},
      {QNN_OP_ELEMENT_WISE_EQUAL, QnnOpCode::kElementWiseEqual},
      {QNN_OP_ELEMENT_WISE_EXP, QnnOpCode::kElementWiseExp},
      {QNN_OP_ELEMENT_WISE_FLOOR, QnnOpCode::kElementWiseFloor},
      {QNN_OP_ELEMENT_WISE_FLOOR_DIV, QnnOpCode::kElementWiseFloorDiv},
      {QNN_OP_ELEMENT_WISE_FMOD, QnnOpCode::kElementWiseFmod},
      {QNN_OP_ELEMENT_WISE_GREATER, QnnOpCode::kElementWiseGreater},
      {QNN_OP_ELEMENT_WISE_GREATER_EQUAL, QnnOpCode::kElementWiseGreaterEqual},
      {QNN_OP_ELEMENT_WISE_LESS, QnnOpCode::kElementWiseLess},
      {QNN_OP_ELEMENT_WISE_LESS_EQUAL, QnnOpCode::kElementWiseLessEqual},
      {QNN_OP_ELEMENT_WISE_LOG, QnnOpCode::kElementWiseLog},
      {QNN_OP_ELEMENT_WISE_MAXIMUM, QnnOpCode::kElementWiseMaximum},
      {QNN_OP_ELEMENT_WISE_MINIMUM, QnnOpCode::kElementWiseMinimum},
      {QNN_OP_ELEMENT_WISE_MOD, QnnOpCode::kElementWiseMod},
      {QNN_OP_ELEMENT_WISE_MULTIPLY, QnnOpCode::kElementWiseMultiply},
      {QNN_OP_ELEMENT_WISE_NEG, QnnOpCode::kElementWiseNeg},
      {QNN_OP_ELEMENT_WISE_NEURON, QnnOpCode::kElementWiseNeuron},
      {QNN_OP_ELEMENT_WISE_NOT, QnnOpCode::kElementWiseNot},
      {QNN_OP_ELEMENT_WISE_NOT_EQUAL, QnnOpCode::kElementWiseNotEqual},
      {QNN_OP_ELEMENT_WISE_OR, QnnOpCode::kElementWiseOr},
      {QNN_OP_ELEMENT_WISE_POWER, QnnOpCode::kElementWisePower},
      {QNN_OP_ELEMENT_WISE_ROUND, QnnOpCode::kElementWiseRound},
      {QNN_OP_ELEMENT_WISE_RSQRT, QnnOpCode::kElementWiseRsqrt},
      {QNN_OP_ELEMENT_WISE_SELECT, QnnOpCode::kElementWiseSelect},
      {QNN_OP_ELEMENT_WISE_SIN, QnnOpCode::kElementWiseSin},
      {QNN_OP_ELEMENT_WISE_SIGN, QnnOpCode::kElementWiseSign},
      {QNN_OP_ELEMENT_WISE_SOFTPLUS, QnnOpCode::kElementWiseSoftplus},
      {QNN_OP_ELEMENT_WISE_SQUARED_DIFFERENCE,
       QnnOpCode::kElementWiseSquaredDifference},
      {QNN_OP_ELEMENT_WISE_SQUARE_ROOT, QnnOpCode::kElementWiseSquareRoot},
      {QNN_OP_ELEMENT_WISE_SUBTRACT, QnnOpCode::kElementWiseSubtract},
      {QNN_OP_ELEMENT_WISE_UNARY, QnnOpCode::kElementWiseUnary},
      {QNN_OP_ELEMENT_WISE_XOR, QnnOpCode::kElementWiseXor},
      {QNN_OP_ELU, QnnOpCode::kElu},
      {QNN_OP_EXPAND_DIMS, QnnOpCode::kExpandDims},
      {QNN_OP_EXTRACT_GLIMPSE, QnnOpCode::kExtractGlimpse},
      {QNN_OP_EXTRACT_PATCHES, QnnOpCode::kExtractPatches},
      {QNN_OP_FULLY_CONNECTED, QnnOpCode::kFullyConnected},
      {QNN_OP_GATHER, QnnOpCode::kGather},
      {QNN_OP_GATHER_ELEMENTS, QnnOpCode::kGatherElements},
      {QNN_OP_GATHER_ND, QnnOpCode::kGatherNd},
      {QNN_OP_GELU, QnnOpCode::kGelu},
      {QNN_OP_GENERATE_PROPOSALS, QnnOpCode::kGenerateProposals},
      {QNN_OP_GET_SPARSE_INDICES, QnnOpCode::kGetSparseIndices},
      {QNN_OP_GET_SPARSE_VALUES, QnnOpCode::kGetSparseValues},
      {QNN_OP_GRID_SAMPLE, QnnOpCode::kGridSample},
      {QNN_OP_GROUP_NORM, QnnOpCode::kGroupNorm},
      {QNN_OP_GRU, QnnOpCode::kGru},
      {QNN_OP_HARD_SWISH, QnnOpCode::kHardSwish},
      {QNN_OP_HEAT_MAP_MAX_KEY_POINT, QnnOpCode::kHeatMapMaxKeyPoint},
      {QNN_OP_IM2_COL, QnnOpCode::kIm2Col},
      {QNN_OP_IF, QnnOpCode::kIf},
      {QNN_OP_IMAGE_PROJECTION_TRANSFORM, QnnOpCode::kImageProjectionTransform},
      {QNN_OP_INSTANCE_NORM, QnnOpCode::kInstanceNorm},
      {QNN_OP_L2_NORM, QnnOpCode::kL2Norm},
      {QNN_OP_L2_POOL_2D, QnnOpCode::kL2Pool2d},
      {QNN_OP_LAYER_NORM, QnnOpCode::kLayerNorm},
      {QNN_OP_LOG_SOFTMAX, QnnOpCode::kLogSoftmax},
      {QNN_OP_LRN, QnnOpCode::kLrn},
      {QNN_OP_LSTM, QnnOpCode::kLstm},
      {QNN_OP_MASKED_SOFTMAX, QnnOpCode::kMaskedSoftmax},
      {QNN_OP_MOMENTS, QnnOpCode::kMoments},
      {QNN_OP_MULTI_CLASS_NMS, QnnOpCode::kMultiClassNms},
      {QNN_OP_NON_MAX_SUPPRESSION, QnnOpCode::kNonMaxSuppression},
      {QNN_OP_NON_ZERO, QnnOpCode::kNonZero},
      {QNN_OP_NV12_TO_RGB, QnnOpCode::kNv12ToRgb},
      {QNN_OP_NV21_TO_RGB, QnnOpCode::kNv21ToRgb},
      {QNN_OP_ONE_HOT, QnnOpCode::kOneHot},
      {QNN_OP_PACK, QnnOpCode::kPack},
      {QNN_OP_MAT_MUL, QnnOpCode::kMatMul},
      {QNN_OP_PAD, QnnOpCode::kPad},
      {QNN_OP_POOL_AVG_2D, QnnOpCode::kPoolAvg2d},
      {QNN_OP_POOL_AVG_3D, QnnOpCode::kPoolAvg3d},
      {QNN_OP_POOL_MAX_2D, QnnOpCode::kPoolMax2d},
      {QNN_OP_POOL_MAX_3D, QnnOpCode::kPoolMax3d},
      {QNN_OP_PRELU, QnnOpCode::kPrelu},
      {QNN_OP_QUANTIZE, QnnOpCode::kQuantize},
      {QNN_OP_REDUCE_MAX, QnnOpCode::kReduceMax},
      {QNN_OP_REDUCE_MEAN, QnnOpCode::kReduceMean},
      {QNN_OP_REDUCE_MIN, QnnOpCode::kReduceMin},
      {QNN_OP_REDUCE_PROD, QnnOpCode::kReduceProd},
      {QNN_OP_REDUCE_SUM, QnnOpCode::kReduceSum},
      {QNN_OP_REDUCE_SUM_SQUARE, QnnOpCode::kReduceSumSquare},
      {QNN_OP_RELU, QnnOpCode::kRelu},
      {QNN_OP_RELU1, QnnOpCode::kRelu1},
      {QNN_OP_RELU6, QnnOpCode::kRelu6},
      {QNN_OP_RELU_MIN_MAX, QnnOpCode::kReluMinMax},
      {QNN_OP_RESHAPE, QnnOpCode::kReshape},
      {QNN_OP_RESIZE, QnnOpCode::kResize},
      {QNN_OP_RESIZE_BILINEAR, QnnOpCode::kResizeBilinear},
      {QNN_OP_RESIZE_NEAREST_NEIGHBOR, QnnOpCode::kResizeNearestNeighbor},
      {QNN_OP_RMS_NORM, QnnOpCode::kRmsNorm},
      {QNN_OP_ROI_ALIGN, QnnOpCode::kRoiAlign},
      {QNN_OP_ROI_POOLING, QnnOpCode::kRoiPooling},
      {QNN_OP_SCATTER_ELEMENTS, QnnOpCode::kScatterElements},
      {QNN_OP_SCATTER_ND, QnnOpCode::kScatterNd},
      {QNN_OP_SHAPE, QnnOpCode::kShape},
      {QNN_OP_SIGMOID, QnnOpCode::kSigmoid},
      {QNN_OP_SOFTMAX, QnnOpCode::kSoftmax},
      {QNN_OP_SPACE_TO_BATCH, QnnOpCode::kSpaceToBatch},
      {QNN_OP_SPACE_TO_DEPTH, QnnOpCode::kSpaceToDepth},
      {QNN_OP_SPARSE_TO_DENSE, QnnOpCode::kSparseToDense},
      {QNN_OP_SPLIT, QnnOpCode::kSplit},
      {QNN_OP_SQUEEZE, QnnOpCode::kSqueeze},
      {QNN_OP_STRIDED_SLICE, QnnOpCode::kStridedSlice},
      {QNN_OP_TANH, QnnOpCode::kTanh},
      {QNN_OP_TILE, QnnOpCode::kTile},
      {QNN_OP_TOP_K, QnnOpCode::kTopK},
      {QNN_OP_TRANSPOSE, QnnOpCode::kTranspose},
      {QNN_OP_TRANSPOSE_CONV_1D, QnnOpCode::kTransposeConv1d},
      {QNN_OP_TRANSPOSE_CONV_2D, QnnOpCode::kTransposeConv2d},
      {QNN_OP_TRANSPOSE_CONV_3D, QnnOpCode::kTransposeConv3d},
      {QNN_OP_UN_PACK, QnnOpCode::kUnPack},
  };

  const auto op_count = ops.size();
  const auto name = "op_type_" + std::string(op_type) + "_op_count_" +
                    std::to_string(op_count);

  return ops.emplace_back(std::move(name), op_type, code_type_map.at(op_type));
}

OpWrapper& CreateSimpleActivationOp(std::vector<OpWrapper>& ops,
                                    const char* op_type,
                                    const TensorWrapper& input_tensor,
                                    const TensorWrapper& output_tensor) {
  auto& ret = CreateOpWrapper(ops, op_type);
  ret.AddInputTensor(input_tensor);
  ret.AddOutputTensor(output_tensor);
  return ret;
}

TensorWrapper& ReplaceOutputTensorForFusedActivation(
    TensorPool& tensor_pool, const uint32_t fused_activation_function,
    std::vector<TensorWrapperRef>& output_tensors) {
  if (fused_activation_function == FusedActivationNone) {
    return output_tensors[0];
  }

  if (output_tensors.size() != 1) {
    QNN_LOG_WARNING(
        "Fused activation function: %d is not None but the size of output "
        "tensors is not 1.",
        fused_activation_function);
  }

  TensorWrapper& activation_input =
      tensor_pool.CloneNativeTensorFrom(output_tensors[0]);
  TensorWrapper& activation_output = output_tensors[0].get();
  output_tensors[0] = TensorWrapperRef(activation_input);
  return activation_output;
}

void AddFusedActivationNode(std::vector<OpWrapper>& res,
                            const uint32_t fused_activation_function,
                            const TensorWrapper& input_tensor,
                            const TensorWrapper& output_tensor) {
  switch (fused_activation_function) {
    case FusedActivationNone: {
      break;
    }
    case FusedActivationRelu: {
      CreateSimpleActivationOp(res, QNN_OP_RELU, input_tensor, output_tensor);
      break;
    }
    case FusedActivationReluN1To1: {
      auto& activation_op = CreateOpWrapper(res, QNN_OP_RELU_MIN_MAX);
      activation_op.AddInputTensor(input_tensor);
      activation_op.AddOutputTensor(output_tensor);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                          -1);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                          1);
      break;
    }
    case FusedActivationRelu6: {
      auto& activation_op = CreateOpWrapper(res, QNN_OP_RELU_MIN_MAX);
      activation_op.AddInputTensor(input_tensor);
      activation_op.AddOutputTensor(output_tensor);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                          0);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                          6);
      break;
    }
    case FusedActivationTanh: {
      CreateSimpleActivationOp(res, QNN_OP_TANH, input_tensor, output_tensor);
      break;
    }
    default: {
      QNN_LOG_WARNING("Unsupported fused activation function: %d",
                      fused_activation_function);
      break;
    }
  }
}

}  // namespace qnn
