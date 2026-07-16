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

#include "ml_drift_delegate/tflite/convert/convert_conv.h"

#include <algorithm>
#include <utility>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

bool TryConvertConvToFullyConnected(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteConvParams* params,
    const ::ml_drift::Convolution2DAttributes& attr,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const ::ml_drift::ir::IrTensorId input_tensor_id = tensor_map[input_id];
  const ::ml_drift::BHWC input_shape =
      ir_model.tensor(input_tensor_id)->desc.GetBHWCShape();

  if (input_shape.w != 1 || input_shape.h != 1 ||
      !::ml_drift::IsConvEquivalentToFullyConnected(attr)) {
    return false;
  }

  ::ml_drift::ir::IrOp* fc_op = ir_model.add_op();
  fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED);
  ir_model.AddConsumer(input_tensor_id, fc_op->id);

  if (!tflite::IsConstantTensor(context.tensors + node.inputs->data[1])) {
    ir_model.AddConsumer(tensor_map[node.inputs->data[1]], fc_op->id);
  }
  const bool has_bias =
      node.inputs->size > 2 && node.inputs->data[2] != kTfLiteOptionalTensor;
  if (has_bias &&
      !tflite::IsConstantTensor(context.tensors + node.inputs->data[2])) {
    ir_model.AddConsumer(tensor_map[node.inputs->data[2]], fc_op->id);
  }

  HandleFusedActivation(params->activation, ir_model, fc_op, tensor_map,
                        node.outputs->data[0]);

  ::ml_drift::FullyConnectedAttributes fc_attr;
  fc_attr.weights = std::move(::ml_drift::GetFloatWeights(attr));
  fc_attr.bias = std::move(attr.bias);
  fc_op->attr = std::move(fc_attr);
  return true;
}

void ResolveGroupedConvolution(
    int input_id, const TfLiteNode& node, const TfLiteConvParams* params,
    ::ml_drift::Convolution2DAttributes attr, int src_group_size,
    int dst_group_size,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  // Resolve grouped convolution into Split -> Convs -> Concat
  ::ml_drift::ir::IrOp* split_op = ir_model.add_op();
  split_op->name = ToString(::ml_drift::OperationType::SPLIT);
  ::ml_drift::SplitAttributes split_attr;
  split_attr.axis = ::ml_drift::Axis::CHANNELS;
  split_op->attr = std::move(split_attr);
  ir_model.AddConsumer(tensor_map[input_id], split_op->id);

  // Create intermediate tensors for Split outputs
  const int split_outputs_count = attr.groups;
  std::vector<::ml_drift::ir::IrTensorId> split_output_ids(
      split_outputs_count);
  const ::ml_drift::ir::IrTensor& src_tensor =
      *ir_model.tensor(tensor_map[input_id]);
  for (int i = 0; i < split_outputs_count; ++i) {
    ::ml_drift::BHWDC split_shape = src_tensor.desc.GetBHWDCShape();
    split_shape.c = src_group_size;
    ::ml_drift::ir::IrTensor* split_out =
        ir_model.add_tensor(src_tensor.desc.GetDataType(), split_shape);
    ir_model.SetProducer(split_out->id, split_op->id);
    split_output_ids[i] = split_out->id;
  }

  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>&
      orig_weights = GetFloatWeights(attr);

  std::vector<::ml_drift::ir::IrTensorId> conv_output_ids(
      split_outputs_count);
  for (int i = 0; i < split_outputs_count; ++i) {
    ::ml_drift::ir::IrOp* grp_conv_op = ir_model.add_op();
    grp_conv_op->name = ToString(::ml_drift::OperationType::CONVOLUTION_2D);

    ::ml_drift::Convolution2DAttributes grp_conv_attr = attr;
    grp_conv_attr.groups = 1;

    auto& grp_weights = grp_conv_attr.weights.emplace<
        ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
    grp_weights.shape = ::ml_drift::OHWI(dst_group_size, orig_weights.shape.h,
                                         orig_weights.shape.w, src_group_size);
    grp_weights.data.resize(grp_weights.shape.DimensionsProduct() +
                            XNN_EXTRA_BYTES / sizeof(float));

    for (int out_i = 0; out_i < dst_group_size; ++out_i) {
      for (int in_i = 0; in_i < src_group_size; ++in_i) {
        for (int ky = 0; ky < orig_weights.shape.h; ++ky) {
          for (int kx = 0; kx < orig_weights.shape.w; ++kx) {
            const int src_index = orig_weights.shape.LinearIndex(
                {i * dst_group_size + out_i, ky, kx, in_i});
            const int dst_index =
                grp_weights.shape.LinearIndex({out_i, ky, kx, in_i});
            grp_weights.Set(dst_index, orig_weights.Get(src_index));
          }
        }
      }
    }

    // Extract bias slice
    grp_conv_attr.bias.shape.v = dst_group_size;
    grp_conv_attr.bias.data.resize(dst_group_size);
    for (int out_i = 0; out_i < dst_group_size; ++out_i) {
      if (i * dst_group_size + out_i < attr.bias.data.size()) {
        grp_conv_attr.bias.data[out_i] =
            attr.bias.data[i * dst_group_size + out_i];
      } else {
        grp_conv_attr.bias.data[out_i] = 0.0f;
      }
    }

    ir_model.AddConsumer(split_output_ids[i], grp_conv_op->id);

    ::ml_drift::BHWC conv_shape =
        CalculateOutputShape(src_tensor.desc.GetBHWCShape(), grp_conv_attr);
    ::ml_drift::ir::IrTensor* conv_out =
        ir_model.add_tensor(src_tensor.desc.GetDataType(), conv_shape);

    grp_conv_op->attr = std::move(grp_conv_attr);
    ir_model.SetProducer(conv_out->id, grp_conv_op->id);
    conv_output_ids[i] = conv_out->id;
  }

  ::ml_drift::ir::IrOp* concat_op = ir_model.add_op();
  concat_op->name = ToString(::ml_drift::OperationType::CONCAT);
  ::ml_drift::ConcatAttributes concat_attr;
  concat_attr.axis = ::ml_drift::Axis::CHANNELS;
  concat_op->attr = std::move(concat_attr);
  for (int i = 0; i < split_outputs_count; ++i) {
    ir_model.AddConsumer(conv_output_ids[i], concat_op->id);
  }
  HandleFusedActivation(params->activation, ir_model, concat_op, tensor_map,
                        node.outputs->data[0]);
}

}  // namespace

void ConvertConv(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::Convolution2DAttributes attr;
  const int input_id = node.inputs->data[0];

  const TfLiteTensor* weights_tensor = context.tensors + node.inputs->data[1];
  if (tflite::IsConstantTensor(weights_tensor)) {
    if (weights_tensor->type == kTfLiteInt4) {
      auto& weights = attr.weights.emplace<
          ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>>();
      PopulateTensor(weights_tensor, node.inputs->data[1], &weights,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights, &attr.scale,
                     &attr.zero_point);
    } else if (weights_tensor->type == kTfLiteInt8) {
      auto& weights = attr.weights.emplace<
          ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>>();
      PopulateTensor(weights_tensor, node.inputs->data[1], &weights,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights, &attr.scale,
                     &attr.zero_point);
    } else {
      auto& weights = attr.weights.emplace<::ml_drift::Tensor<
          ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
      PopulateTensor(weights_tensor, node.inputs->data[1], &weights,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights);
    }
    const ::ml_drift::BHWC src_shape =
        ir_model.tensor(tensor_map[input_id])->desc.GetBHWCShape();
    attr.groups =
        src_shape.c /
        std::visit([](const auto& w) { return w.shape; }, attr.weights).i;
  } else {
    auto& weights = attr.weights.emplace<
        ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
    const ::ml_drift::ir::IrTensorId weights_id =
        tensor_map[node.inputs->data[1]];
    const ::ml_drift::BHWC weights_shape =
        ir_model.tensor(weights_id)->desc.GetBHWCShape();
    // For runtime weights, TFLite conv2d weights are OHWI.
    weights.shape = ::ml_drift::OHWI(weights_shape.b, weights_shape.h,
                                     weights_shape.w, weights_shape.c);
    attr.groups = 1;
  }

  const bool has_bias =
      node.inputs->size > 2 && node.inputs->data[2] != kTfLiteOptionalTensor;
  if (has_bias &&
      tflite::IsConstantTensor(context.tensors + node.inputs->data[2])) {
    PopulateTensor(context.tensors + node.inputs->data[2], node.inputs->data[2],
                   &attr.bias, PopulateTensorFlags::kNoExtraBytes,
                   options.enable_spanned_weights);
  }

  const auto* params = static_cast<const TfLiteConvParams*>(node.builtin_data);
  attr.strides = ToHW(params->stride_height, params->stride_width);
  attr.dilations = ::ml_drift::HW(std::max(1, params->dilation_height_factor),
                                  std::max(1, params->dilation_width_factor));

  UpdatePadding(params->padding,
                ir_model.tensor(tensor_map[input_id])->desc.GetBHWDCShape(),
                &attr);
  const int src_group_size =
      std::visit([](const auto& w) { return w.shape; }, attr.weights).i;
  const int dst_group_size =
      std::visit([](const auto& w) { return w.shape; }, attr.weights).o /
      attr.groups;
  const bool supported_grouped_conv =
      src_group_size % 4 == 0 && dst_group_size % 4 == 0;

  if (TryConvertConvToFullyConnected(context, node, params, attr, tensor_map,
                                     ir_model)) {
    return;
  }

  if (attr.groups != 1 && !supported_grouped_conv) {
    ResolveGroupedConvolution(input_id, node, params, std::move(attr),
                              src_group_size, dst_group_size, tensor_map,
                              ir_model);
  } else {
    ::ml_drift::ir::IrOp* conv_op = ir_model.add_op();
    conv_op->name = ToString(::ml_drift::OperationType::CONVOLUTION_2D);
    ir_model.AddConsumer(tensor_map[input_id], conv_op->id);
    if (!tflite::IsConstantTensor(context.tensors + node.inputs->data[1])) {
      ir_model.AddConsumer(tensor_map[node.inputs->data[1]], conv_op->id);
    }
    if (has_bias &&
        !tflite::IsConstantTensor(context.tensors + node.inputs->data[2])) {
      ir_model.AddConsumer(tensor_map[node.inputs->data[2]], conv_op->id);
    }

    HandleFusedActivation(params->activation, ir_model, conv_op, tensor_map,
                          node.outputs->data[0]);
    conv_op->attr = std::move(attr);
  }
}

}  // namespace litert::ml_drift::ir
