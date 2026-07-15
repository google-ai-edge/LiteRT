// Copyright 2024 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/tflite/support/support.h"

#include <cstring>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support_absolute_positional_embedding.h"
#include "ml_drift_delegate/tflite/support/support_argmax.h"
#include "ml_drift_delegate/tflite/support/support_batch_matmul.h"
#include "ml_drift_delegate/tflite/support/support_bitcast.h"
#include "ml_drift_delegate/tflite/support/support_broadcast_in_dim.h"
#include "ml_drift_delegate/tflite/support/support_clamp.h"
#include "ml_drift_delegate/tflite/support/support_concat.h"
#include "ml_drift_delegate/tflite/support/support_conv.h"
#include "ml_drift_delegate/tflite/support/support_cumsum.h"
#include "ml_drift_delegate/tflite/support/support_depth_to_space.h"
#include "ml_drift_delegate/tflite/support/support_depthwise_conv.h"
#include "ml_drift_delegate/tflite/support/support_dequantize.h"
#include "ml_drift_delegate/tflite/support/support_dynamic_update_slice.h"
#include "ml_drift_delegate/tflite/support/support_elementwise.h"
#include "ml_drift_delegate/tflite/support/support_embedding_lookup.h"
#include "ml_drift_delegate/tflite/support/support_fully_connected.h"
#include "ml_drift_delegate/tflite/support/support_gather.h"
#include "ml_drift_delegate/tflite/support/support_group_norm.h"
#include "ml_drift_delegate/tflite/support/support_layer_norm.h"
#include "ml_drift_delegate/tflite/support/support_one_hot.h"
#include "ml_drift_delegate/tflite/support/support_pack.h"
#include "ml_drift_delegate/tflite/support/support_pad.h"
#include "ml_drift_delegate/tflite/support/support_pixel_shuffle.h"
#include "ml_drift_delegate/tflite/support/support_pooling2d.h"
#include "ml_drift_delegate/tflite/support/support_prelu.h"
#include "ml_drift_delegate/tflite/support/support_quantize.h"
#include "ml_drift_delegate/tflite/support/support_reduce.h"
#include "ml_drift_delegate/tflite/support/support_relu.h"
#include "ml_drift_delegate/tflite/support/support_resampler.h"
#include "ml_drift_delegate/tflite/support/support_reshape.h"
#include "ml_drift_delegate/tflite/support/support_resize2d.h"
#include "ml_drift_delegate/tflite/support/support_reverse.h"
#include "ml_drift_delegate/tflite/support/support_rms_norm.h"
#include "ml_drift_delegate/tflite/support/support_rotary_positional_embedding.h"
#include "ml_drift_delegate/tflite/support/support_sdpa.h"
#include "ml_drift_delegate/tflite/support/support_select.h"
#include "ml_drift_delegate/tflite/support/support_slice.h"
#include "ml_drift_delegate/tflite/support/support_softmax.h"
#include "ml_drift_delegate/tflite/support/support_space_to_depth.h"
#include "ml_drift_delegate/tflite/support/support_split.h"
#include "ml_drift_delegate/tflite/support/support_splitv.h"
#include "ml_drift_delegate/tflite/support/support_strided_slice.h"
#include "ml_drift_delegate/tflite/support/support_tile.h"
#include "ml_drift_delegate/tflite/support/support_topk.h"
#include "ml_drift_delegate/tflite/support/support_transpose.h"
#include "ml_drift_delegate/tflite/support/support_transpose_conv.h"
#include "ml_drift_delegate/tflite/support/support_unpack.h"
#include "ml_drift_delegate/tflite/support/support_unpooling2d.h"
// copybara:uncomment_begin(google-only)
// #include "ml_drift_delegate/tflite/support/google/support_alignment_points_to_transform_matrix.h"
// #include "ml_drift_delegate/tflite/support/google/support_keep_if_max_2d_pt2.h"
// #include "ml_drift_delegate/tflite/support/google/support_landmarks_to_transform_matrix.h"
// #include "ml_drift_delegate/tflite/support/google/support_roi_to_transform_matrix.h"
// #include "ml_drift_delegate/tflite/support/google/support_transform_landmarks.h"
// #include "ml_drift_delegate/tflite/support/google/support_transform_tensor_bilinear.h"
// copybara:uncomment_end
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils.h"
#include "tflite/util.h"

namespace litert::ml_drift::ir {
namespace {

bool IsCompositeNodeSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error, const CustomIrOpMap* custom_parsers) {
  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
  if (params == nullptr) {
    *error = "Missing StableHLO composite params.";
    return false;
  }

  if (custom_parsers) {
    if (auto it = custom_parsers->find(params->name);
        it != custom_parsers->end()) {
      if (it->second.is_supported(context, node, registration).ok()) {
        return true;
      }
      *error = absl::StrCat("Custom parser ", params->name, " rejected node.");
      return false;
    }
  }

  if (std::strcmp(params->name, "odml.group_norm") == 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (flexbuffer_map["_TENSOR_V1_reduction_axes"].IsNull()) {
      return IsLayerNormSupported(context, node, registration, error);
    } else {
      return IsGroupNormSupported(context, node, registration, error);
    }
  } else if (std::strcmp(params->name, "odml.rms_norm") == 0) {
    return IsRmsNormSupported(context, node, registration, error);
  } else if (std::strcmp(params->name, "odml.scaled_dot_product_attention") ==
             0) {
    return IsSdpaSupported(context, node, registration, error);
  } else {
    *error = absl::StrCat("Unknown StableHLO composite params: ", params->name);
    return false;
  }
}

bool IsCustomNodeSupported(const TfLiteContext* absl_nonnull context,
                           const TfLiteNode* absl_nonnull node,
                           const TfLiteRegistration* absl_nonnull registration,
                           std::string* absl_nonnull error,
                           const CustomIrOpMap* custom_parsers) {
  const absl::string_view custom_name = registration->custom_name;
  if (custom_parsers) {
    auto it = custom_parsers->find(custom_name);
    if (it != custom_parsers->end()) {
      if (it->second.is_supported(context, node, registration).ok()) {
        return true;
      }
      *error = absl::StrCat("Custom parser ", custom_name, " rejected node.");
      return false;
    }
  }
  if (custom_name == "Convolution2DTransposeBias") {
    return IsTransposeConvSupported(context, node, registration, error);
  } else if (custom_name == "custom_call.absolute_positional_embedding") {
    return IsAbsolutePositionalEmbeddingSupported(context, node, registration,
                                                  error);
  } else if (custom_name == "custom_call.pixel_shuffle") {
    return IsPixelShuffleSupported(context, node, registration, error);
  } else if (custom_name == "custom_call.rotary_positional_embedding") {
    return IsRotaryPositionalEmbeddingSupported(context, node, registration,
                                                error);
  } else if (custom_name == "custom_call.MaxUnpooling2D") {
    return IsUnpooling2dSupported(context, node, registration, error);
  } else if (custom_name == "MaxPoolingWithArgmax2D") {
    return IsPooling2dSupported(context, node, registration, error);
  } else if (custom_name == "Resampler") {
    return IsResamplerSupported(context, node, registration, error);
  }
  // copybara:uncomment_begin(google-only)
  // if (custom_name == "AlignmentPointsToTransformMatrix") {
    // return IsAlignmentPointsToTransformMatrixSupported(context, node,
                                                       // registration, error);
  // } else if (custom_name == "KeepIfMax2D") {
    // return IsKeepIfMax2dPt2Supported(context, node, registration, error);
  // } else if (custom_name == "Landmarks2TransformMatrix" ||
             // custom_name == "Landmarks2TransformMatrixV2") {
    // return IsLandmarksToTransformMatrixSupported(context, node, registration,
                                                 // error);
  // } else if (custom_name == "RoIToTransformMatrix") {
    // return IsRoiToTransformMatrixSupported(context, node, registration, error);
  // } else if (custom_name == "TransformLandmarks") {
    // return IsTransformLandmarksSupported(context, node, registration, error);
  // } else if (custom_name == "TransformTensor" ||
             // custom_name == "TransformTensorBilinear") {
    // return IsTransformTensorBilinearSupported(context, node, registration,
                                              // error);
  // }
  // copybara:uncomment_end
  return false;
}

bool IsNodeSupported(const TfLiteContext* absl_nonnull context,
                     const TfLiteNode* absl_nonnull node,
                     const TfLiteRegistration* absl_nonnull registration,
                     std::string* absl_nonnull error,
                     const CustomIrOpMap* custom_parsers) {
  // Convenience lambdas for readability of the switch-case block.
  auto argmax =
      absl::bind_front(IsArgMaxSupported, context, node, registration, error);
  auto arith1 = [context, node, registration,
                 error](int supported_max_version) {
    return IsUnaryArithmeticOpSupported(context, node, registration,
                                        supported_max_version, error);
  };
  auto arith2 = [context, node, registration,
                 error](int supported_max_version) {
    return IsBinaryArithmeticOpSupported(context, node, registration,
                                         supported_max_version, error);
  };
  auto bmm = absl::bind_front(IsBatchMatMulSupported, context, node,
                              registration, error);
  auto broadcast_in_dim = absl::bind_front(IsBroadcastInDimSupported, context,
                                           node, registration, error);
  auto bitcast =
      absl::bind_front(IsBitcastSupported, context, node, registration, error);
  auto clamp =
      absl::bind_front(IsClampSupported, context, node, registration, error);
  auto composite = [context, node, registration, error, custom_parsers]() {
    return IsCompositeNodeSupported(context, node, registration, error,
                                    custom_parsers);
  };
  auto concat =
      absl::bind_front(IsConcatSupported, context, node, registration, error);
  auto conv2d =
      absl::bind_front(IsConv2dSupported, context, node, registration, error);
  auto cumsum =
      absl::bind_front(IsCumsumSupported, context, node, registration, error);
  auto custom = [context, node, registration, error, custom_parsers]() {
    return IsCustomNodeSupported(context, node, registration, error,
                                 custom_parsers);
  };
  auto depthwise_conv2d = absl::bind_front(IsDepthwiseConv2dSupported, context,
                                           node, registration, error);
  auto depth_to_space = absl::bind_front(IsDepthToSpaceSupported, context, node,
                                         registration, error);
  auto dequantize = absl::bind_front(IsDequantizeSupported, context, node,
                                     registration, error);
  auto dynamic_update_slice = absl::bind_front(
      IsDynamicUpdateSliceSupported, context, node, registration, error);
  auto fully_connected = absl::bind_front(IsFullyConnectedSupported, context,
                                          node, registration, error);
  auto gather =
      absl::bind_front(IsGatherSupported, context, node, registration, error);
  auto embedding_lookup = absl::bind_front(IsEmbeddingLookupSupported, context,
                                           node, registration, error);
  auto logic1 = [context, node, registration,
                 error](int supported_max_version) {
    return IsUnaryLogicalOpSupported(context, node, registration,
                                     supported_max_version, error);
  };
  auto logic2 = [context, node, registration,
                 error](int supported_max_version) {
    return IsBinaryLogicalOpSupported(context, node, registration,
                                      supported_max_version, error);
  };
  auto lstm = [=](int supported_max_version) {
    return false;
    // TODO(b/438520776): Re-enable once LSTM convert is live.
    // return IsLstmSupported(context, node, registration,
    // supported_max_version,
    //                        error);
  };
  auto one_hot =
      absl::bind_front(IsOneHotSupported, context, node, registration, error);
  auto pack =
      absl::bind_front(IsPackSupported, context, node, registration, error);
  auto pad =
      absl::bind_front(IsPadSupported, context, node, registration, error);
  auto pooling2d = absl::bind_front(IsPooling2dSupported, context, node,
                                    registration, error);
  auto prelu = [context, node, registration, error](int supported_max_version) {
    return IsPReLUSupported(context, node, registration, supported_max_version,
                            error);
  };
  auto quantize =
      absl::bind_front(IsQuantizeSupported, context, node, registration, error);
  auto reduce =
      absl::bind_front(IsReduceSupported, context, node, registration, error);
  auto relu = [context, node, registration, error](int supported_max_version) {
    return IsReluSupported(context, node, registration, supported_max_version,
                           error);
  };
  auto reshape =
      absl::bind_front(IsReshapeSupported, context, node, registration, error);
  auto resize2d =
      absl::bind_front(IsResize2DSupported, context, node, registration, error);
  auto reverse =
      absl::bind_front(IsReverseSupported, context, node, registration, error);
  auto select =
      absl::bind_front(IsSelectSupported, context, node, registration, error);
  auto slice =
      absl::bind_front(IsSliceSupported, context, node, registration, error);
  auto softmax = [context, node, registration,
                  error](int supported_max_version) {
    return IsSoftmaxSupported(context, node, registration,
                              supported_max_version, error);
  };
  auto space_to_depth = absl::bind_front(IsSpaceToDepthSupported, context, node,
                                         registration, error);
  auto split =
      absl::bind_front(IsSplitSupported, context, node, registration, error);
  auto splitv =
      absl::bind_front(IsSplitVSupported, context, node, registration, error);
  auto strided_slice = absl::bind_front(IsStridedSliceSupported, context, node,
                                        registration, error);
  auto tile =
      absl::bind_front(IsTileSupported, context, node, registration, error);
  auto top_k =
      absl::bind_front(IsTopKSupported, context, node, registration, error);
  auto transpose = absl::bind_front(IsTransposeSupported, context, node,
                                    registration, error);
  auto unpack =
      absl::bind_front(IsUnpackSupported, context, node, registration, error);
  auto transpose_conv = absl::bind_front(IsTransposeConvSupported, context,
                                         node, registration, error);

  // clang-format off
  switch (registration->builtin_code) {
    // go/keep-sorted start
    case kTfLiteBuiltinAbs:                     return arith1(5);
    case kTfLiteBuiltinAdd:                     return arith2(6);
    case kTfLiteBuiltinArgMax:                  return argmax();
    case kTfLiteBuiltinAtan2:                   return arith2(2);
    case kTfLiteBuiltinAveragePool2d:           return pooling2d();
    case kTfLiteBuiltinBatchMatmul:             return bmm();
    case kTfLiteBuiltinBitcast:                 return bitcast();
    case kTfLiteBuiltinBitwiseXor:              return logic2(2);
    case kTfLiteBuiltinCast:                    return arith1(1);
    case kTfLiteBuiltinCeil:                    return arith1(2);
    case kTfLiteBuiltinConcatenation:           return concat();
    case kTfLiteBuiltinConv2d:                  return conv2d();
    case kTfLiteBuiltinCos:                     return arith1(2);
    case kTfLiteBuiltinCumsum:                  return cumsum();
    case kTfLiteBuiltinCustom:                  return custom();
    case kTfLiteBuiltinDepthToSpace:            return depth_to_space();
    case kTfLiteBuiltinDepthwiseConv2d:         return depthwise_conv2d();
    case kTfLiteBuiltinDequantize:              return dequantize();
    case kTfLiteBuiltinDiv:                     return arith2(2);
    case kTfLiteBuiltinDynamicUpdateSlice:      return dynamic_update_slice();
    case kTfLiteBuiltinElu:                     return arith1(2);
    case kTfLiteBuiltinEmbeddingLookup:         return embedding_lookup();
    case kTfLiteBuiltinEqual:                   return logic2(2);
    case kTfLiteBuiltinExp:                     return arith1(2);
    case kTfLiteBuiltinFloor:                   return arith1(2);
    case kTfLiteBuiltinFloorDiv:                return arith2(3);
    case kTfLiteBuiltinFloorMod:                return arith2(2);
    case kTfLiteBuiltinFullyConnected:          return fully_connected();
    case kTfLiteBuiltinGather:                  return gather();
    case kTfLiteBuiltinGelu:                    return arith1(3);
    case kTfLiteBuiltinGreater:                 return logic2(2);
    case kTfLiteBuiltinGreaterEqual:            return logic2(2);
    case kTfLiteBuiltinHardSwish:               return arith1(1);
    case kTfLiteBuiltinLeakyRelu:               return relu(2);
    case kTfLiteBuiltinLess:                    return logic2(2);
    case kTfLiteBuiltinLessEqual:               return logic2(2);
    case kTfLiteBuiltinLog:                     return arith1(2);
    case kTfLiteBuiltinLogicalAnd:              return logic2(2);
    case kTfLiteBuiltinLogicalNot:              return logic1(2);
    case kTfLiteBuiltinLogicalOr:               return logic2(2);
    case kTfLiteBuiltinLogistic:                return arith1(2);
    case kTfLiteBuiltinLstm:                    return lstm(4);
    case kTfLiteBuiltinMaxPool2d:               return pooling2d();
    case kTfLiteBuiltinMaximum:                 return arith2(4);
    case kTfLiteBuiltinMean:                    return reduce();
    case kTfLiteBuiltinMinimum:                 return arith2(4);
    case kTfLiteBuiltinMirrorPad:               return pad();
    case kTfLiteBuiltinMul:                     return arith2(8);
    case kTfLiteBuiltinNeg:                     return arith1(2);
    case kTfLiteBuiltinNotEqual:                return logic2(2);
    case kTfLiteBuiltinOneHot:                  return one_hot();
    case kTfLiteBuiltinPack:                    return pack();
    case kTfLiteBuiltinPad:                     return pad();
    case kTfLiteBuiltinPadv2:                   return pad();
    case kTfLiteBuiltinPow:                     return arith2(2);
    case kTfLiteBuiltinPrelu:                   return prelu(1);
    case kTfLiteBuiltinQuantize:                return quantize();
    case kTfLiteBuiltinReduceAll:               return reduce();
    case kTfLiteBuiltinReduceAny:               return reduce();
    case kTfLiteBuiltinReduceMax:               return reduce();
    case kTfLiteBuiltinReduceMin:               return reduce();
    case kTfLiteBuiltinReduceProd:              return reduce();
    case kTfLiteBuiltinRelu6:                   return relu(2);
    case kTfLiteBuiltinRelu:                    return relu(2);
    case kTfLiteBuiltinReluN1To1:               return relu(2);
    case kTfLiteBuiltinReshape:                 return reshape();
    case kTfLiteBuiltinResizeBilinear:          return resize2d();
    case kTfLiteBuiltinResizeNearestNeighbor:   return resize2d();
    case kTfLiteBuiltinReverseV2:               return reverse();
    case kTfLiteBuiltinRightShift:              return arith2(1);
    case kTfLiteBuiltinRound:                   return arith1(2);
    case kTfLiteBuiltinRsqrt:                   return arith1(2);
    case kTfLiteBuiltinSelect:                  return select();
    case kTfLiteBuiltinSelectV2:                return select();
    case kTfLiteBuiltinSign:                    return arith1(2);
    case kTfLiteBuiltinSin:                     return arith1(2);
    case kTfLiteBuiltinSlice:                   return slice();
    case kTfLiteBuiltinSoftmax:                 return softmax(4);
    case kTfLiteBuiltinSpaceToDepth:            return space_to_depth();
    case kTfLiteBuiltinSplit:                   return split();
    case kTfLiteBuiltinSplitV:                  return splitv();
    case kTfLiteBuiltinSqrt:                    return arith1(2);
    case kTfLiteBuiltinSquare:                  return arith1(2);
    case kTfLiteBuiltinSquaredDifference:       return arith2(2);
    case kTfLiteBuiltinStablehloBroadcastInDim: return broadcast_in_dim();
    case kTfLiteBuiltinStablehloCbrt:           return arith1(1);
    case kTfLiteBuiltinStablehloClamp:          return clamp();
    case kTfLiteBuiltinStablehloComposite:      return composite();
    case kTfLiteBuiltinStablehloRemainder:      return arith2(1);
    case kTfLiteBuiltinStablehloShiftLeft:      return arith2(1);
    case kTfLiteBuiltinStridedSlice:            return strided_slice();
    case kTfLiteBuiltinSub:                     return arith2(3);
    case kTfLiteBuiltinSum:                     return reduce();
    case kTfLiteBuiltinTanh:                    return arith1(2);
    case kTfLiteBuiltinTile:                    return tile();
    case kTfLiteBuiltinTopkV2:                  return top_k();
    case kTfLiteBuiltinTranspose:               return transpose();
    case kTfLiteBuiltinTransposeConv:           return transpose_conv();
    case kTfLiteBuiltinUnpack:                  return unpack();
    // go/keep-sorted end
    default:                                    return false;
  }
  // clang-format on
}

bool HasBoolTensor(const TfLiteContext* absl_nonnull context,
                   const TfLiteNode* absl_nonnull node) {
  for (int i = 0; i < node->inputs->size; ++i) {
    const int tensor_id = node->inputs->data[i];
    if (tensor_id == kTfLiteOptionalTensor) continue;
    if (context->tensors[tensor_id].type == kTfLiteBool) return true;
  }
  for (int i = 0; i < node->outputs->size; ++i) {
    const int tensor_id = node->outputs->data[i];
    if (tensor_id == kTfLiteOptionalTensor) continue;
    if (context->tensors[tensor_id].type == kTfLiteBool) return true;
  }
  return false;
}

bool HasQuantTensor(const TfLiteContext* absl_nonnull context,
                    const TfLiteNode* absl_nonnull node,
                    const TfLiteRegistration* absl_nonnull registration) {
  const std::unordered_set<TfLiteType> quant_types = {
      kTfLiteInt2, kTfLiteInt4, kTfLiteInt8, kTfLiteUInt4, kTfLiteUInt8};
  for (int i = 0; i < node->inputs->size; ++i) {
    const int tensor_id = node->inputs->data[i];
    if (tensor_id == kTfLiteOptionalTensor) continue;
    if (quant_types.contains(context->tensors[tensor_id].type)) return true;
  }
  for (int i = 0; i < node->outputs->size; ++i) {
    const int tensor_id = node->outputs->data[i];
    if (tensor_id == kTfLiteOptionalTensor) continue;
    if (quant_types.contains(context->tensors[tensor_id].type)) return true;
  }
  if (registration->builtin_code == kTfLiteBuiltinQuantize) return true;
  if (registration->builtin_code == kTfLiteBuiltinDequantize) return true;
  return false;
}

}  // namespace

std::vector<int> GetSupportedNodes(TfLiteContext* absl_nonnull context,
                                   const IrModelBuilderOptions& options) {
  return GetSupportedNodes(context, options, nullptr);
}

// Runs GraphPartitionHelper with the above IsNodeSupported.
std::vector<int> GetSupportedNodes(TfLiteContext* absl_nonnull context,
                                   const IrModelBuilderOptions& options,
                                   const CustomIrOpMap* custom_parsers) {
  auto is_node_supported_wrapper = [&](const TfLiteContext* context,
                                       const TfLiteNode* node,
                                       const TfLiteRegistration* registration,
                                       std::string* error) {
    if (!options.allow_bool_tensors && HasBoolTensor(context, node)) {
      *error = "Op has bool tensors which are not supported.";
      return false;
    }
    if (!options.allow_quant_ops &&
        HasQuantTensor(context, node, registration)) {
      *error = "Op has quant tensors which are not supported.";
      return false;
    }
    return IsNodeSupported(context, node, registration, error, custom_parsers);
  };

  ::tflite::delegates::FP16GraphPartitionHelper partitioner(
      context, is_node_supported_wrapper);
  std::set<std::string> errors;
  std::vector<int> supported_ops;
  if (partitioner.Partition(&errors, options.start_node_index,
                            options.end_node_index) == kTfLiteOk) {
    supported_ops = partitioner.GetNodesOfFirstNLargestPartitions(
        options.max_delegated_partitions);
  }
  if (!errors.empty() && partitioner.num_total_nodes() > supported_ops.size()) {
    std::string joined_errors = "\n  Not supported by ML Drift:\n";
    for (const auto& error : errors) {
      absl::StrAppend(&joined_errors, "    ", error, "\n");
    }
    ABSL_LOG(INFO) << joined_errors;
  }
  return supported_ops;
}

TfLiteIntArray* GetOpsToReplace(TfLiteContext* absl_nonnull context,
                                const IrModelBuilderOptions& options,
                                const CustomIrOpMap* custom_parsers) {
  int all_ops;
  {
    TfLiteIntArray* exec_plan;
    ABSL_QCHECK_EQ(context->GetExecutionPlan(context, &exec_plan), kTfLiteOk);
    ABSL_QCHECK_NE(exec_plan, nullptr);
    all_ops = exec_plan->size;
  }
  const auto supported_ops =
      GetSupportedNodes(context, options, custom_parsers);
  ABSL_LOG_IF(INFO, supported_ops.size() < all_ops) << absl::StrCat(
      supported_ops.size(), " / ", all_ops, " ops are delegated to ML Drift.");
  return ::tflite::ConvertVectorToTfLiteIntArray(supported_ops);
}

}  // namespace litert::ml_drift::ir
