// Copyright 2026 Google LLC.
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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_fully_connected.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

namespace {

void SetFullyConnectedOutputShape(const ::ml_drift::BHWC& input_shape,
                                  int output_channels,
                                  ::ml_drift::BHWC& expected_output_shape) {
  expected_output_shape = input_shape;
  expected_output_shape.c = output_channels;
}

bool IsFullyConnectedOutputReshapeNeeded(const ::ml_drift::BHWC& input_shape,
                                         int output_channels,
                                         const ::ml_drift::BHWC& output_shape) {
  ::ml_drift::BHWC expected_output_shape;
  SetFullyConnectedOutputShape(input_shape, output_channels,
                               expected_output_shape);
  return output_shape != expected_output_shape;
}

template <typename AttrType>
void PopulateQuantizedAttributes(const TfLiteTensor* weights_tensor,
                                 int weights_id,
                                 const TfLiteTensor* bias_tensor, int bias_id,
                                 bool bias_is_const,
                                 bool enable_spanned_weights, AttrType& attr) {
  ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::INT8> weights;
  PopulateTensor(weights_tensor, weights_id, &weights,
                 PopulateTensorFlags::kExtraBytes, enable_spanned_weights,
                 &attr.scale, &attr.zero_point);

  int num_elements = weights.shape.DimensionsProduct();
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8> fc_weights;
  fc_weights.spanned_data = std::move(weights.spanned_data);
  fc_weights.id = weights.id;
  fc_weights.shape.o = weights.shape.h;
  fc_weights.shape.h = 1;
  fc_weights.shape.w = 1;
  fc_weights.shape.i = weights.shape.w;

  // Have to unpack int2/int4 weights into int8 for RearrangeWeights
  // Once we have RearrangeWeightsPacked supported for all layouts, we can
  // remove this unpacking.
  if (weights_tensor->type == kTfLiteInt4) {
    std::vector<int8_t> unpacked_data(num_elements);
    ::ml_drift::UnpackDenseInt4IntoInt8(weights.data.data(), num_elements,
                                        unpacked_data.data());
    fc_weights.data = std::move(unpacked_data);
  } else if (weights_tensor->type == kTfLiteInt2) {
    std::vector<int8_t> unpacked_data(num_elements);
    ::ml_drift::UnpackDenseInt2IntoInt8(weights.data.data(), num_elements,
                                        unpacked_data.data());
    fc_weights.data = std::move(unpacked_data);
  } else {
    fc_weights.data = std::move(weights.data);
  }

  attr.weights = std::move(fc_weights);

  if (bias_is_const) {
    PopulateTensor(bias_tensor, bias_id, &attr.bias,
                   PopulateTensorFlags::kNoExtraBytes, enable_spanned_weights);
  }
}

}  // namespace

void ConvertFullyConnected(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int weights_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];

  const TfLiteTensor* weights_tensor = context.tensors + weights_id;
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node.builtin_data);

  ::ml_drift::ir::IrOp* fc_op = ir_model.add_op();
  ir_model.AddConsumer(tensor_map[input_id], fc_op->id);

  const bool weights_are_const = tflite::IsConstantTensor(weights_tensor);
  const bool has_bias =
      node.inputs->size > 2 && node.inputs->data[2] != kTfLiteOptionalTensor;
  const TfLiteTensor* bias_tensor =
      has_bias ? (context.tensors + node.inputs->data[2]) : nullptr;
  const bool bias_is_const = has_bias && tflite::IsConstantTensor(bias_tensor);
  int bias_id = -1;
  if (has_bias) {
    bias_id = node.inputs->data[2];
  }

  const ::ml_drift::BHWC input_shape =
      ir_model.tensor(tensor_map[input_id])->desc.GetBHWCShape();
  const ::ml_drift::BHWC output_shape =
      ir_model.tensor(tensor_map[output_id])->desc.GetBHWCShape();

  if (weights_are_const) {
    if (weights_tensor->type == kTfLiteInt8 ||
        weights_tensor->type == kTfLiteInt4 ||
        weights_tensor->type == kTfLiteInt2) {
      if (weights_tensor->type == kTfLiteInt8) {
        ::ml_drift::FullyConnectedInt8Attributes attr;
        PopulateQuantizedAttributes(weights_tensor, weights_id, bias_tensor,
                                    bias_id, bias_is_const,
                                    options.enable_spanned_weights, attr);
        fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8);
        fc_op->attr = std::move(attr);
      } else if (weights_tensor->type == kTfLiteInt4) {
        ::ml_drift::FullyConnectedInt4Attributes attr;
        PopulateQuantizedAttributes(weights_tensor, weights_id, bias_tensor,
                                    bias_id, bias_is_const,
                                    options.enable_spanned_weights, attr);
        fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT4);
        fc_op->attr = std::move(attr);
      } else if (weights_tensor->type == kTfLiteInt2) {
        ::ml_drift::FullyConnectedInt2Attributes attr;
        PopulateQuantizedAttributes(weights_tensor, weights_id, bias_tensor,
                                    bias_id, bias_is_const,
                                    options.enable_spanned_weights, attr);
        fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT2);
        fc_op->attr = std::move(attr);
      }
    } else {
      ::ml_drift::FullyConnectedAttributes attr;
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32> weights;
      PopulateTensor(weights_tensor, weights_id, &weights,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights);
      attr.weights.data = std::move(weights.data);
      attr.weights.spanned_data = std::move(weights.spanned_data);
      attr.weights.id = weights.id;
      attr.weights.shape.h = 1;
      attr.weights.shape.w = 1;
      attr.weights.shape.o = weights.shape.h;
      attr.weights.shape.i = weights.shape.w;

      if (bias_is_const) {
        PopulateTensor(bias_tensor, node.inputs->data[2], &attr.bias,
                       PopulateTensorFlags::kNoExtraBytes,
                       options.enable_spanned_weights);
      }

      if (input_shape.h != 1 || input_shape.w != 1) {
        ::ml_drift::Convolution2DAttributes conv_attr;
        conv_attr.strides = ::ml_drift::HW(1, 1);
        conv_attr.dilations = ::ml_drift::HW(1, 1);
        conv_attr.padding.appended = ::ml_drift::HW(0, 0);
        conv_attr.padding.prepended = ::ml_drift::HW(0, 0);
        conv_attr.weights = attr.weights;
        conv_attr.bias = attr.bias;
        fc_op->name = ToString(::ml_drift::OperationType::CONVOLUTION_2D);
        fc_op->attr = std::move(conv_attr);
      } else {
        fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED);
        fc_op->attr = std::move(attr);
      }
    }
    if (has_bias && !bias_is_const) {
      ir_model.AddConsumer(tensor_map[node.inputs->data[2]], fc_op->id);
    }
  } else {
    fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED);

    auto* weights_ir_tensor = ir_model.tensor(tensor_map[weights_id]);
    const auto& current_shape = weights_ir_tensor->desc.GetBHWCShape();

    // While we check if const weights are (o, 1, 1, i) sh, we don't for
    // non-const weights. Manually add a reshape here if necessary.
    if (current_shape.h != 1 || current_shape.w != 1) {
      ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
      reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
      ir_model.AddConsumer(tensor_map[weights_id], reshape_op->id);
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape =
          ::ml_drift::BHWC(output_shape.c, 1, 1, input_shape.c);
      reshape_op->attr = std::move(reshape_attr);

      const ::ml_drift::BHWDC target_shape_bhwdc(output_shape.c, 1, 1, 1,
                                                 input_shape.c);
      const ::ml_drift::ir::IrTensor* intermediate = ir_model.add_tensor(
          weights_ir_tensor->desc.GetDataType(), target_shape_bhwdc);
      ir_model.SetProducer(intermediate->id, reshape_op->id);
      ir_model.AddConsumer(intermediate->id, fc_op->id);
    } else {
      // If it's already 1x1, just use it directly
      ir_model.AddConsumer(tensor_map[weights_id], fc_op->id);
    }

    if (has_bias) {
      ir_model.AddConsumer(tensor_map[node.inputs->data[2]], fc_op->id);
    }
    ::ml_drift::FullyConnectedAttributes attr;
    fc_op->attr = std::move(attr);
  }

  const ::ml_drift::BHWDC weights_shape_bhwdc =
      ExtractTensorShape(weights_tensor->dims);
  const ::ml_drift::BHWC weights_shape(
      weights_shape_bhwdc.b, weights_shape_bhwdc.h, weights_shape_bhwdc.w,
      weights_shape_bhwdc.c);
  const bool reshape_needed = IsFullyConnectedOutputReshapeNeeded(
      input_shape, weights_tensor->dims->data[0], output_shape);

  if (reshape_needed) {
    ::ml_drift::BHWC expected_shape;
    SetFullyConnectedOutputShape(input_shape, weights_tensor->dims->data[0],
                                 expected_shape);
    const ::ml_drift::ir::IrTensor* intermediate = ir_model.add_tensor(
        ir_model.tensor(tensor_map[output_id])->desc.GetDataType(),
        expected_shape);
    ir_model.SetProducer(intermediate->id, fc_op->id);

    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    ir_model.AddConsumer(intermediate->id, reshape_op->id);
    ::ml_drift::ReshapeAttributes reshape_attr;
    reshape_attr.new_shape = output_shape;
    reshape_op->attr = std::move(reshape_attr);

    HandleFusedActivation(params->activation, ir_model, reshape_op, tensor_map,
                          output_id);
  } else {
    HandleFusedActivation(params->activation, ir_model, fc_op, tensor_map,
                          output_id);
  }
}

}  // namespace litert::ml_drift::ir
