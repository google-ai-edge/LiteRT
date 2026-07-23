// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/ir_model_builder.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/transformations/ir/transform.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_absolute_positional_embedding.h"
#include "ml_drift_delegate/tflite/convert/convert_argmax.h"
#include "ml_drift_delegate/tflite/convert/convert_batch_matmul.h"
#include "ml_drift_delegate/tflite/convert/convert_bitcast.h"
#include "ml_drift_delegate/tflite/convert/convert_broadcast_in_dim.h"
#include "ml_drift_delegate/tflite/convert/convert_cbrt.h"
#include "ml_drift_delegate/tflite/convert/convert_clamp.h"
#include "ml_drift_delegate/tflite/convert/convert_concat.h"
#include "ml_drift_delegate/tflite/convert/convert_conv.h"
#include "ml_drift_delegate/tflite/convert/convert_cumsum.h"
#include "ml_drift_delegate/tflite/convert/convert_depth_to_space.h"
#include "ml_drift_delegate/tflite/convert/convert_depthwise_conv.h"
#include "ml_drift_delegate/tflite/convert/convert_dequantize.h"
#include "ml_drift_delegate/tflite/convert/convert_dynamic_update_slice.h"
#include "ml_drift_delegate/tflite/convert/convert_elementwise.h"
#include "ml_drift_delegate/tflite/convert/convert_embedding_lookup.h"
#include "ml_drift_delegate/tflite/convert/convert_fully_connected.h"
#include "ml_drift_delegate/tflite/convert/convert_gather.h"
#include "ml_drift_delegate/tflite/convert/convert_group_norm.h"
#include "ml_drift_delegate/tflite/convert/convert_layer_norm.h"
#include "ml_drift_delegate/tflite/convert/convert_one_hot.h"
#include "ml_drift_delegate/tflite/convert/convert_pack.h"
#include "ml_drift_delegate/tflite/convert/convert_pad.h"
#include "ml_drift_delegate/tflite/convert/convert_pixel_shuffle.h"
#include "ml_drift_delegate/tflite/convert/convert_pooling2d.h"
#include "ml_drift_delegate/tflite/convert/convert_prelu.h"
#include "ml_drift_delegate/tflite/convert/convert_quantize.h"
#include "ml_drift_delegate/tflite/convert/convert_reduce.h"
#include "ml_drift_delegate/tflite/convert/convert_relu.h"
#include "ml_drift_delegate/tflite/convert/convert_resampler.h"
#include "ml_drift_delegate/tflite/convert/convert_reshape.h"
#include "ml_drift_delegate/tflite/convert/convert_resize2d.h"
#include "ml_drift_delegate/tflite/convert/convert_reverse.h"
#include "ml_drift_delegate/tflite/convert/convert_rms_norm.h"
#include "ml_drift_delegate/tflite/convert/convert_rope.h"
#include "ml_drift_delegate/tflite/convert/convert_sdpa.h"
#include "ml_drift_delegate/tflite/convert/convert_select.h"
#include "ml_drift_delegate/tflite/convert/convert_slice.h"
#include "ml_drift_delegate/tflite/convert/convert_softmax.h"
#include "ml_drift_delegate/tflite/convert/convert_space_to_depth.h"
#include "ml_drift_delegate/tflite/convert/convert_split.h"
#include "ml_drift_delegate/tflite/convert/convert_strided_slice.h"
#include "ml_drift_delegate/tflite/convert/convert_tile.h"
#include "ml_drift_delegate/tflite/convert/convert_topk.h"
#include "ml_drift_delegate/tflite/convert/convert_transpose.h"
#include "ml_drift_delegate/tflite/convert/convert_transpose_conv.h"
#include "ml_drift_delegate/tflite/convert/convert_unpack.h"
#include "ml_drift_delegate/tflite/convert/convert_unpooling2d.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
// copybara:uncomment_begin(google-only)
// #include "ml_drift_delegate/tflite/convert/google/convert_alignment_points_to_transform_matrix.h"
// #include "ml_drift_delegate/tflite/convert/google/convert_keep_if_max_2d_pt2.h"
// #include "ml_drift_delegate/tflite/convert/google/convert_landmarks_to_transform_matrix.h"
// #include "ml_drift_delegate/tflite/convert/google/convert_roi_to_transform_matrix.h"
// #include "ml_drift_delegate/tflite/convert/google/convert_transform_landmarks.h"
// #include "ml_drift_delegate/tflite/convert/google/convert_transform_tensor_bilinear.h"
// copybara:uncomment_end
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "ml_drift_delegate/tflite/support/support.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/subgraph.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift::ir {
namespace {

// Converts a custom TFLite node and appends it to the `ir_model`.
// TODO: dlho - Add to custom op
void ConvertCustom(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, const CustomIrOpMap* custom_parsers,
    ::ml_drift::ir::IrModel& ir_model) {
  const absl::string_view custom_name = registration.custom_name;
  if (custom_parsers) {
    auto it = custom_parsers->find(custom_name);
    if (it != custom_parsers->end()) {
      it->second.convert(context, node, registration, tensor_map, options,
                         ir_model);
      return;
    }
  }
  if (custom_name == "Convolution2DTransposeBias") {
    ConvertTransposeConv(context, node, registration, tensor_map, options,
                         ir_model);
    return;
  } else if (custom_name == "custom_call.pixel_shuffle") {
    ConvertPixelShuffle(context, node, registration, tensor_map, ir_model);
    return;
  } else if (custom_name == "custom_call.absolute_positional_embedding") {
    ConvertAbsolutePositionalEmbedding(context, node, registration, tensor_map,
                                       ir_model);
    return;
  } else if (custom_name == "custom_call.rotary_positional_embedding") {
    ConvertRoPE(context, node, registration, tensor_map, ir_model);
    return;
  } else if (custom_name == "MaxPoolingWithArgmax2D") {
    ConvertPooling2d(context, node, registration, tensor_map, ir_model);
    return;
  } else if (custom_name == "custom_call.MaxUnpooling2D") {
    ConvertUnpooling2d(context, node, registration, tensor_map, ir_model);
    return;
  } else if (custom_name == "Resampler") {
    ConvertResampler(context, node, registration, tensor_map, ir_model);
    return;
  }
  // copybara:uncomment_begin(google-only)
  // if (custom_name == "AlignmentPointsToTransformMatrix") {
    // ConvertAlignmentPointsToTransformMatrix(context, node, registration,
                                            // tensor_map, ir_model);
    // return;
  // } else if (custom_name == "KeepIfMax2D") {
    // ConvertKeepIfMax2dPt2(context, node, registration, tensor_map, ir_model);
    // return;
  // } else if (custom_name == "Landmarks2TransformMatrix" ||
             // custom_name == "Landmarks2TransformMatrixV2") {
    // ConvertLandmarksToTransformMatrix(context, node, registration, tensor_map,
                                      // ir_model);
    // return;
  // } else if (custom_name == "RoIToTransformMatrix") {
    // ConvertRoiToTransformMatrix(context, node, registration, tensor_map,
                                // ir_model);
    // return;
  // } else if (custom_name == "TransformLandmarks") {
    // ConvertTransformLandmarks(context, node, registration, tensor_map,
                              // ir_model);
    // return;
  // } else if (custom_name == "TransformTensor" ||
             // custom_name == "TransformTensorBilinear") {
    // ConvertTransformTensorBilinear(context, node, registration, tensor_map,
                                   // ir_model);
    // return;
  // }
  // copybara:uncomment_end
  ABSL_LOG(FATAL) << "Unsupported custom op: " << custom_name;
}

// Converts a StableHLO composite node and appends it to the `ir_model`.
void ConvertComposite(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, const CustomIrOpMap* custom_parsers,
    ::ml_drift::ir::IrModel& ir_model) {
  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node.builtin_data);
  if (!params) {
    ABSL_LOG(FATAL) << "Missing StableHLO composite params.";
  }
  const absl::string_view composite_name = params->name;

  if (custom_parsers) {
    const auto it = custom_parsers->find(composite_name);
    if (it != custom_parsers->end()) {
      it->second.convert(context, node, registration, tensor_map, options,
                         ir_model);
      return;
    }
  }

  if (composite_name == "odml.group_norm") {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (flexbuffer_map["_TENSOR_V1_reduction_axes"].IsNull()) {
      ConvertLayerNorm(context, node, registration, tensor_map, ir_model);
      return;
    } else {
      ConvertGroupNorm(context, node, registration, tensor_map, ir_model);
      return;
    }
  } else if (composite_name == "odml.rms_norm") {
    ConvertRmsNorm(context, node, registration, tensor_map, ir_model);
    return;
  } else if (composite_name == "odml.scaled_dot_product_attention") {
    ConvertSdpa(context, node, registration, tensor_map, ir_model);
    return;
  }
  ABSL_LOG(FATAL) << "Unsupported composite op: " << composite_name;
}

// Converts a TensorFlow Lite subgraph into an ML Drift-internal Intermediate
// Representation (IR) model.
class IrModelBuilder {
 public:
  // Constructs the IrModelBuilder.
  IrModelBuilder(
      const TfLiteContext& context, const TfLiteDelegateParams& delegate_params,
      const IrModelBuilderOptions& options,
      const CustomIrOpMap* custom_parsers = nullptr,
      SharedConstTensorsMap* shared_tensors = nullptr,
      const TensorIndexToBufferIdMap* tensor_to_shared_buffer_id_map = nullptr,
      const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
          nullptr)
      : context_(context),
        delegate_params_(delegate_params),
        options_(options),
        custom_parsers_(custom_parsers),
        shared_tensors_(shared_tensors),
        tensor_to_shared_buffer_id_map_(tensor_to_shared_buffer_id_map),
        tensor_to_external_buffer_id_map_(tensor_to_external_buffer_id_map) {}

  // Executes the conversion from TFLite model to ML Drift IrModel.
  ::ml_drift::ir::IrModel* Build() {
    auto ir_model = std::make_unique<::ml_drift::ir::IrModel>();

    // Ensures that when we create IR ops, all their corresponding input/output
    // IR tensors are already available for lookup. Note that some of these
    // tensors may become unused after graph transformations like op fusion.
    auto tensor_map = CreateTensorMap(*ir_model);

    for (int i = 0; i < delegate_params_.input_tensors->size; ++i) {
      const int input_tensor_id = delegate_params_.input_tensors->data[i];
      if (::tflite::IsConstantTensor(&context_.tensors[input_tensor_id])) {
        continue;
      }
      ir_model->add_input(tensor_map[input_tensor_id]);
    }
    for (int i = 0; i < delegate_params_.output_tensors->size; ++i) {
      ir_model->add_output(
          tensor_map[delegate_params_.output_tensors->data[i]]);
    }

    // Convert each TFLite node to its IR equivalent and append it to the
    // `ir_model`. This stage may introduce new nodes and tensors not
    // originally present in the TFLite model (e.g., from op decomposition).
    const TfLiteIntArray* nodes = delegate_params_.nodes_to_replace;
    for (int node_id = 0; node_id < nodes->size; ++node_id) {
      const auto& [node, registration] = GetNodeInfo(node_id);
      // Skip f16 dequantize ops if no other nodes precede them.
      // Note the corresponding change in support_dequantize.cc.
      if (registration->builtin_code == kTfLiteBuiltinDequantize &&
          context_.tensors[node->inputs->data[0]].type ==
              TfLiteType::kTfLiteFloat16 &&
          ::tflite::IsConstantTensor(
              &context_.tensors[node->inputs->data[0]])) {
        continue;
      }
      AddNode(*node, *registration, tensor_map, *ir_model);
    }
    // Derive the shared-constants map from per-tensor BufferSource state
    // populated during graph construction and enriched by op converters (e.g.
    // dequant_forced). Done after conversion so op-provided fields are
    // included. The tflite tensor id comes from the tensor map key.
    if (shared_tensors_) {
      for (const auto& [tfl_tensor_id, ir_tensor_id] : tensor_map) {
        const auto* tensor = ir_model->tensor(ir_tensor_id);
        if (tensor == nullptr || !tensor->buffer_source.is_shared) {
          continue;
        }
        SharedTfliteTensor shared_info;
        shared_info.tflite_tensor_id = tfl_tensor_id;
        shared_info.global_id = tensor->buffer_source.global_id;
        shared_info.dequant_forced = tensor->buffer_source.dequant_forced;
        if (tensor->buffer_source.force_linear_layout) {
          shared_info.layout = ::ml_drift::Layout::LINEAR;
        }
        shared_tensors_->try_emplace(ir_tensor_id, shared_info);
      }
    }
    return ir_model.release();
  }

 private:
  // Converts a single TFLite node and appends it to the `ir_model`.
  void AddNode(const TfLiteNode& node, const TfLiteRegistration& registration,
               absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
               ::ml_drift::ir::IrModel& ir_model) const {
    auto argmax = absl::bind_front(ConvertArgMax, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto batch_matmul =
        absl::bind_front(ConvertBatchMatMul, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto bitcast =
        absl::bind_front(ConvertBitcast, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto broadcast_in_dim =
        absl::bind_front(ConvertBroadcastInDim, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto cbrt = absl::bind_front(ConvertCbrt, context_, node, registration,
                                 std::ref(tensor_map), std::ref(ir_model));
    auto clamp = absl::bind_front(ConvertClamp, context_, node, registration,
                                  std::ref(tensor_map), std::ref(ir_model));
    auto composite = absl::bind_front(
        ConvertComposite, context_, node, registration, std::ref(tensor_map),
        std::ref(options_), custom_parsers_, std::ref(ir_model));
    auto concat = absl::bind_front(ConvertConcat, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto conv =
        absl::bind_front(ConvertConv, context_, node, registration,
                         std::ref(tensor_map), options_, std::ref(ir_model));
    auto cumsum = absl::bind_front(ConvertCumsum, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto custom = absl::bind_front(ConvertCustom, context_, node, registration,
                                   std::ref(tensor_map), options_,
                                   custom_parsers_, std::ref(ir_model));
    auto depth_to_space =
        absl::bind_front(ConvertDepthToSpace, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto depthwise_conv =
        absl::bind_front(ConvertDepthwiseConv, context_, node, registration,
                         std::ref(tensor_map), options_, std::ref(ir_model));
    auto dequantize =
        absl::bind_front(ConvertDequantize, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto dyn_update_slice = absl::bind_front(
        ConvertDynamicUpdateSlice, context_, node, registration,
        std::ref(tensor_map), std::ref(ir_model));
    auto elementwise =
        absl::bind_front(ConvertElementwise, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto embedding_lookup =
        absl::bind_front(ConvertEmbeddingLookup, context_, node, registration,
                         std::ref(tensor_map), options_, std::ref(ir_model));
    auto fully_connected =
        absl::bind_front(ConvertFullyConnected, context_, node, registration,
                         std::ref(tensor_map), options_, std::ref(ir_model));
    auto gather = absl::bind_front(ConvertGather, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto one_hot = absl::bind_front(ConvertOneHot, context_, node, registration,
                                    std::ref(tensor_map), std::ref(ir_model));
    auto pack = absl::bind_front(ConvertPack, context_, node, registration,
                                 std::ref(tensor_map), std::ref(ir_model));
    auto pad =
        absl::bind_front(ConvertPad, context_, node, registration,
                         std::ref(tensor_map), options_, std::ref(ir_model));
    auto pooling2d =
        absl::bind_front(ConvertPooling2d, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto prelu = absl::bind_front(ConvertPrelu, context_, node, registration,
                                  std::ref(tensor_map), std::ref(ir_model));
    auto quantize =
        absl::bind_front(ConvertQuantize, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto reduce = absl::bind_front(ConvertReduce, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto relu = absl::bind_front(ConvertRelu, node, registration,
                                 std::ref(tensor_map), std::ref(ir_model));
    auto reshape =
        absl::bind_front(ConvertReshape, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto resize2d =
        absl::bind_front(ConvertResize2d, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto reverse =
        absl::bind_front(ConvertReverse, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto select = absl::bind_front(ConvertSelect, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto slice = absl::bind_front(ConvertSlice, context_, node, registration,
                                  std::ref(tensor_map), std::ref(ir_model));
    auto softmax =
        absl::bind_front(ConvertSoftmax, context_, node, registration, options_,
                         std::ref(tensor_map), std::ref(ir_model));
    auto space_to_depth =
        absl::bind_front(ConvertSpaceToDepth, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto split = absl::bind_front(ConvertSplit, context_, node, registration,
                                  std::ref(tensor_map), std::ref(ir_model));
    auto splitv = absl::bind_front(ConvertSplitV, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    auto strided_slice =
        absl::bind_front(ConvertStridedSlice, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto tile = absl::bind_front(ConvertTile, context_, node, registration,
                                 std::ref(tensor_map), std::ref(ir_model));
    auto top_k = absl::bind_front(ConvertTopK, context_, node, registration,
                                  std::ref(tensor_map), std::ref(ir_model));
    auto transpose =
        absl::bind_front(ConvertTranspose, context_, node, registration,
                         std::ref(tensor_map), std::ref(ir_model));
    auto transpose_conv =
        absl::bind_front(ConvertTransposeConv, context_, node, registration,
                         std::ref(tensor_map), options_, std::ref(ir_model));
    auto unpack = absl::bind_front(ConvertUnpack, context_, node, registration,
                                   std::ref(tensor_map), std::ref(ir_model));
    // clang-format off
    switch (registration.builtin_code) {
      // go/keep-sorted start
      case kTfLiteBuiltinAbs:                      return elementwise();
      case kTfLiteBuiltinAdd:                      return elementwise();
      case kTfLiteBuiltinArgMax:                   return argmax();
      case kTfLiteBuiltinAtan2:                    return elementwise();
      case kTfLiteBuiltinAveragePool2d:            return pooling2d();
      case kTfLiteBuiltinBatchMatmul:              return batch_matmul();
      case kTfLiteBuiltinBitcast:                  return bitcast();
      case kTfLiteBuiltinBitwiseXor:               return elementwise();
      case kTfLiteBuiltinCast:                     return elementwise();
      case kTfLiteBuiltinCeil:                     return elementwise();
      case kTfLiteBuiltinConcatenation:            return concat();
      case kTfLiteBuiltinConv2d:                   return conv();
      case kTfLiteBuiltinCos:                      return elementwise();
      case kTfLiteBuiltinCumsum:                   return cumsum();
      case kTfLiteBuiltinCustom:                   return custom();
      case kTfLiteBuiltinDepthToSpace:             return depth_to_space();
      case kTfLiteBuiltinDepthwiseConv2d:          return depthwise_conv();
      case kTfLiteBuiltinDequantize:               return dequantize();
      case kTfLiteBuiltinDiv:                      return elementwise();
      case kTfLiteBuiltinDynamicUpdateSlice:       return dyn_update_slice();
      case kTfLiteBuiltinElu:                      return elementwise();
      case kTfLiteBuiltinEmbeddingLookup:          return embedding_lookup();
      case kTfLiteBuiltinEqual:                    return elementwise();
      case kTfLiteBuiltinExp:                      return elementwise();
      case kTfLiteBuiltinFloor:                    return elementwise();
      case kTfLiteBuiltinFloorDiv:                 return elementwise();
      case kTfLiteBuiltinFloorMod:                 return elementwise();
      case kTfLiteBuiltinFullyConnected:           return fully_connected();
      case kTfLiteBuiltinGather:                   return gather();
      case kTfLiteBuiltinGelu:                     return elementwise();
      case kTfLiteBuiltinGreater:                  return elementwise();
      case kTfLiteBuiltinGreaterEqual:             return elementwise();
      case kTfLiteBuiltinHardSwish:                return elementwise();
      case kTfLiteBuiltinLeakyRelu:                return relu();
      case kTfLiteBuiltinLess:                     return elementwise();
      case kTfLiteBuiltinLessEqual:                return elementwise();
      case kTfLiteBuiltinLog:                      return elementwise();
      case kTfLiteBuiltinLogicalAnd:               return elementwise();
      case kTfLiteBuiltinLogicalNot:               return elementwise();
      case kTfLiteBuiltinLogicalOr:                return elementwise();
      case kTfLiteBuiltinLogistic:                 return elementwise();
      case kTfLiteBuiltinMaxPool2d:                return pooling2d();
      case kTfLiteBuiltinMaximum:                  return elementwise();
      case kTfLiteBuiltinMean:                     return reduce();
      case kTfLiteBuiltinMinimum:                  return elementwise();
      case kTfLiteBuiltinMirrorPad:                return pad();
      case kTfLiteBuiltinMul:                      return elementwise();
      case kTfLiteBuiltinNeg:                      return elementwise();
      case kTfLiteBuiltinNotEqual:                 return elementwise();
      case kTfLiteBuiltinOneHot:                   return one_hot();
      case kTfLiteBuiltinPack:                     return pack();
      case kTfLiteBuiltinPad:                      return pad();
      case kTfLiteBuiltinPadv2:                    return pad();
      case kTfLiteBuiltinPow:                      return elementwise();
      case kTfLiteBuiltinPrelu:                    return prelu();
      case kTfLiteBuiltinQuantize:                 return quantize();
      case kTfLiteBuiltinReduceAll:                return reduce();
      case kTfLiteBuiltinReduceAny:                return reduce();
      case kTfLiteBuiltinReduceMax:                return reduce();
      case kTfLiteBuiltinReduceMin:                return reduce();
      case kTfLiteBuiltinReduceProd:               return reduce();
      case kTfLiteBuiltinRelu0To1:                 return relu();
      case kTfLiteBuiltinRelu6:                    return relu();
      case kTfLiteBuiltinRelu:                     return relu();
      case kTfLiteBuiltinReluN1To1:                return relu();
      case kTfLiteBuiltinReshape:                  return reshape();
      case kTfLiteBuiltinResizeBilinear:           return resize2d();
      case kTfLiteBuiltinResizeNearestNeighbor:    return resize2d();
      case kTfLiteBuiltinReverseV2:                return reverse();
      case kTfLiteBuiltinRightShift:               return elementwise();
      case kTfLiteBuiltinRound:                    return elementwise();
      case kTfLiteBuiltinRsqrt:                    return elementwise();
      case kTfLiteBuiltinSelect:                   return select();
      case kTfLiteBuiltinSelectV2:                 return select();
      case kTfLiteBuiltinSign:                     return elementwise();
      case kTfLiteBuiltinSin:                      return elementwise();
      case kTfLiteBuiltinSlice:                    return slice();
      case kTfLiteBuiltinSoftmax:                  return softmax();
      case kTfLiteBuiltinSpaceToDepth:             return space_to_depth();
      case kTfLiteBuiltinSplit:                    return split();
      case kTfLiteBuiltinSplitV:                   return splitv();
      case kTfLiteBuiltinSqrt:                     return elementwise();
      case kTfLiteBuiltinSquare:                   return elementwise();
      case kTfLiteBuiltinSquaredDifference:        return elementwise();
      case kTfLiteBuiltinStablehloBroadcastInDim:  return broadcast_in_dim();
      case kTfLiteBuiltinStablehloCbrt:            return cbrt();
      case kTfLiteBuiltinStablehloClamp:           return clamp();
      case kTfLiteBuiltinStablehloComposite:       return composite();
      case kTfLiteBuiltinStablehloRemainder:       return elementwise();
      case kTfLiteBuiltinStablehloShiftLeft:       return elementwise();
      case kTfLiteBuiltinStridedSlice:             return strided_slice();
      case kTfLiteBuiltinSub:                      return elementwise();
      case kTfLiteBuiltinSum:                      return reduce();
      case kTfLiteBuiltinTanh:                     return elementwise();
      case kTfLiteBuiltinTile:                     return tile();
      case kTfLiteBuiltinTopkV2:                   return top_k();
      case kTfLiteBuiltinTranspose:                return transpose();
      case kTfLiteBuiltinTransposeConv:            return transpose_conv();
      case kTfLiteBuiltinUnpack:                   return unpack();
      // go/keep-sorted end

      default:  // Should've filtered in IsSupported.
        ABSL_LOG(FATAL) << "Invalid op code " << registration.builtin_code;
    }
    // clang-format on
  }

  ::ml_drift::ir::IrTensorId AddTensor(
      int tensor_id, ::ml_drift::ir::IrModel& ir_model,
      ::ml_drift::ir::BufferSource buffer_source = {}) const {
    if (tensor_id == kTfLiteOptionalTensor) {
      return ::ml_drift::ir::IrTensorId{
          static_cast<::ml_drift::ir::IrTensorId>(-1)};
    }
    const TfLiteTensor& tflite_tensor = context_.tensors[tensor_id];
    const ::ml_drift::DataType dtype = GetDtype(tflite_tensor.type);
    const ::ml_drift::BHWDC shape = ExtractTensorShape(tflite_tensor.dims);
    auto* tensor = ir_model.add_tensor(dtype, shape);
    tensor->buffer_source = buffer_source;
    return tensor->id;
  }

  // Helper function to create an IR tensor from a TFLite tensor. If the tensor
  // is a constant and its ID is found in the external buffer maps, it is marked
  // as a shared constant and added to the `shared_tensors_` map.
  void ProcessTensor(
      int tfl_tensor_id, ::ml_drift::ir::IrModel& ir_model,
      absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map) const {
    // Skip optional tensors and tensors that have already been processed.
    if (tfl_tensor_id == kTfLiteOptionalTensor ||
        tensor_map.contains(tfl_tensor_id)) {
      return;
    }

    bool is_shared = false;
    int global_id = -1;

    // Check if the tensor is a constant and if its ID is found externally
    if (::tflite::IsConstantTensor(&context_.tensors[tfl_tensor_id])) {
      if (tensor_to_external_buffer_id_map_) {
        if (auto it = tensor_to_external_buffer_id_map_->find(tfl_tensor_id);
            it != tensor_to_external_buffer_id_map_->end()) {
          is_shared = true;
          global_id = it->second;
        }
      }
      if (!is_shared && tensor_to_shared_buffer_id_map_) {
        if (auto it = tensor_to_shared_buffer_id_map_->find(tfl_tensor_id);
            it != tensor_to_shared_buffer_id_map_->end()) {
          is_shared = true;
          global_id = it->second;
        }
      }
    }

    // Add the tensor to the IR model and update the tensor map.
    // Record the shared-buffer metadata on the tensor itself. The shared
    // constants map is derived from this in Build() after op conversion.
    const auto ir_tensor_id = AddTensor(
        tfl_tensor_id, ir_model,
        ::ml_drift::ir::BufferSource{is_shared, global_id});
    tensor_map[tfl_tensor_id] = ir_tensor_id;
  }

  // Creates IR tensors for all unique TFLite tensors in the subgraph.
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> CreateTensorMap(
      ::ml_drift::ir::IrModel& ir_model) const {
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;

    const TfLiteIntArray* nodes = delegate_params_.nodes_to_replace;
    for (int i = 0; i < nodes->size; ++i) {
      const auto& [node, registration] = GetNodeInfo(i);
      for (int j = 0; j < node->inputs->size; ++j) {
        ProcessTensor(node->inputs->data[j], ir_model, tensor_map);
      }
      for (int j = 0; j < node->outputs->size; ++j) {
        ProcessTensor(node->outputs->data[j], ir_model, tensor_map);
      }
    }
    return tensor_map;
  }

  ::ml_drift::DataType GetDtype(TfLiteType tflite_type) const {
    switch (tflite_type) {
      case kTfLiteFloat32:
        return ::ml_drift::DataType::FLOAT32;
      case kTfLiteFloat16:
        return ::ml_drift::DataType::FLOAT16;
      case kTfLiteBFloat16:
        return ::ml_drift::DataType::BFLOAT16;
      case kTfLiteInt2:
        return ::ml_drift::DataType::INT2;
      case kTfLiteInt4:
        return ::ml_drift::DataType::INT4;
      case kTfLiteUInt4:
        return ::ml_drift::DataType::UINT4;
      case kTfLiteInt8:
        return ::ml_drift::DataType::INT8;
      case kTfLiteUInt8:
        return ::ml_drift::DataType::UINT8;
      case kTfLiteInt16:
        return ::ml_drift::DataType::INT16;
      case kTfLiteUInt16:
        return ::ml_drift::DataType::UINT16;
      case kTfLiteInt32:
        return ::ml_drift::DataType::INT32;
      case kTfLiteUInt32:
        return ::ml_drift::DataType::UINT32;
      case kTfLiteBool:
        return ::ml_drift::DataType::BOOL;
      default:
        // Returning UNKNOWN allows the framework to fail gracefully downstream
        // when it attempts to select a GPU kernel.
        return ::ml_drift::DataType::UNKNOWN;
    }
  }

  // C++ wrapper for GetNodeAndRegistration to enable structured bindings, i.e.,
  // const auto& [node, registration] = GetNodeAndRegistration(node_id);
  std::pair<TfLiteNode*, TfLiteRegistration*> GetNodeInfo(int node_id) const {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    context_.GetNodeAndRegistration(
        const_cast<TfLiteContext*>(&context_),
        delegate_params_.nodes_to_replace->data[node_id], &node, &registration);
    return {node, registration};
  }

  const TfLiteContext& context_;
  const TfLiteDelegateParams& delegate_params_;
  const IrModelBuilderOptions& options_;
  const CustomIrOpMap* custom_parsers_;
  SharedConstTensorsMap* shared_tensors_;
  const TensorIndexToBufferIdMap* tensor_to_shared_buffer_id_map_;
  const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map_;
};

struct DelegateData {
  const IrModelBuilderOptions& options;
  ::ml_drift::ir::IrModel* ir_model;
};

class DelegateContext {
 public:
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    const auto* delegate_data =
        reinterpret_cast<const DelegateData*>(delegate_params->delegate->data_);
    std::unique_ptr<::ml_drift::ir::IrModel> built_model(
        BuildIrModel(*context, *delegate_params, delegate_data->options));
    if (!built_model) return false;
    *delegate_data->ir_model = std::move(*built_model);
    return true;
  }
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteRegistration registration = {
      .init = [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        auto* delegate_context = new DelegateContext();
        if (!delegate_context->Init(
                context,
                reinterpret_cast<const TfLiteDelegateParams*>(buffer))) {
          delete delegate_context;
          return nullptr;
        }
        return delegate_context;
      },
      .free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<DelegateContext*>(buffer);
      },
      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      .invoke = nullptr,
      .custom_name = "MlDriftDelegate(IR)",
  };

  const auto* delegate_data =
      reinterpret_cast<const DelegateData*>(delegate->data_);
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, delegate_data->options);
  if (ops_to_replace->size == 0) {
    TfLiteIntArrayFree(ops_to_replace);
    return kTfLiteOk;
  }

  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, registration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace

::ml_drift::ir::IrModel* BuildIrModel(
    const TfLiteContext& context, const TfLiteDelegateParams& delegate_params,
    const IrModelBuilderOptions& options, const CustomIrOpMap* custom_parsers,
    SharedConstTensorsMap* shared_tensors,
    const TensorIndexToBufferIdMap* tensor_to_shared_buffer_id_map,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map) {
  IrModelBuilder builder(context, delegate_params, options, custom_parsers,
                         shared_tensors, tensor_to_shared_buffer_id_map,
                         tensor_to_external_buffer_id_map);
  std::unique_ptr<::ml_drift::ir::IrModel> ir_model(builder.Build());
  if (!ir_model) {
    return nullptr;
  }

  // Transform and optimize the IR graph.
  // This could include op fusion, dead code elimination, layout changes, etc.
  if (options.apply_model_transformations) {
    if (!::ml_drift::ir::TransformIrModel(ir_model.get()).ok()) {
      return nullptr;
    }
  }
  return ir_model.release();
}

absl::Status BuildFromFlatBuffer(const ::tflite::FlatBufferModel& flatbuffer,
                                 const ::tflite::OpResolver& op_resolver,
                                 const IrModelBuilderOptions& options,
                                 ::ml_drift::ir::IrModel* ir_model) {
  std::unique_ptr<::tflite::Interpreter> interpreter;
  ::tflite::InterpreterBuilder interpreter_builder(flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("Unable to prepare TfLite interpreter.");
  }

  TfLiteDelegate delegate = TfLiteDelegateCreate();
  DelegateData delegate_data = {
      .options = options,
      .ir_model = ir_model,
  };
  delegate.data_ = &delegate_data;
  delegate.Prepare = DelegatePrepare;
  delegate.flags = kTfLiteDelegateFlagsNone;

  if (interpreter->ModifyGraphWithDelegate(&delegate) != kTfLiteOk) {
    return absl::InternalError("Conversion from TfLite model failed.");
  }

  // Transform and optimize the IR graph.
  // This could include op fusion, dead code elimination, layout changes, etc.
  if (options.apply_model_transformations) {
    RETURN_IF_ERROR(::ml_drift::ir::TransformIrModel(ir_model));
  }
  return absl::OkStatus();
}

}  // namespace litert::ml_drift::ir
