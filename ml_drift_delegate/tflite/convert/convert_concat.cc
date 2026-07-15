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

#include "ml_drift_delegate/tflite/convert/convert_concat.h"

#include <cstddef>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

bool IsConstantZeros(const TfLiteTensor* tensor) {
  if (!tflite::IsConstantTensor(tensor) || !tensor->data.raw) return false;
  for (size_t j = 0; j < tensor->bytes; ++j) {
    if (tensor->data.raw_const[j] != 0) return false;
  }
  return true;
}

// Tries to convert a CONCAT op with a constant zeros input to a PAD op.
// Returns true if the conversion is successful, false otherwise.
bool TryConvertConcatToPad(
    const TfLiteContext& context, const TfLiteNode& node,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  // First check necessary conditions for conversion.
  if (node.inputs->size != 2) return false;

  int zeros_index = -1;
  for (int i = 0; i < 2; ++i) {
    if (IsConstantZeros(context.tensors + node.inputs->data[i])) {
      zeros_index = i;
      break;
    }
  }
  if (zeros_index == -1) return false;

  std::vector<::ml_drift::BHWDC> input_shapes;
  input_shapes.reserve(2);
  for (int i = 0; i < 2; ++i) {
    input_shapes.push_back(
        ExtractTensorShape(context.tensors[node.inputs->data[i]].dims));
  }
  const ::ml_drift::BHWDC output_shape =
      ir_model.tensor(tensor_map[node.outputs->data[0]])->desc.GetBHWDCShape();

  ::ml_drift::Axis axis = GetConcatAxis(input_shapes, output_shape);
  if (axis != ::ml_drift::Axis::HEIGHT && axis != ::ml_drift::Axis::WIDTH &&
      axis != ::ml_drift::Axis::CHANNELS) {
    return false;
  }

  // Conditions are met, proceed with conversion.
  int non_zero_index = 1 - zeros_index;
  const TfLiteTensor* input_tensor =
      context.tensors + node.inputs->data[non_zero_index];
  ::ml_drift::ir::IrTensorId input_id;
  if (tflite::IsConstantTensor(input_tensor)) {
    input_id =
        AddConstInput(context, node.inputs->data[non_zero_index], ir_model, {})
            ->id;
  } else {
    input_id = tensor_map[node.inputs->data[non_zero_index]];
  }

  ::ml_drift::ir::IrOp* pad_op = ir_model.add_op();
  pad_op->name = ToString(::ml_drift::OperationType::PAD);
  ir_model.AddConsumer(input_id, pad_op->id);

  ::ml_drift::PadAttributes pad_attr;
  pad_attr.type = ::ml_drift::PaddingContentType::ZEROS;
  pad_attr.appended = ::ml_drift::BHWC(0, 0, 0, 0);
  pad_attr.prepended = ::ml_drift::BHWC(0, 0, 0, 0);

  ::ml_drift::BHWDC zero_shape = input_shapes[zeros_index];
  ::ml_drift::BHWC* p =
      (zeros_index == 0) ? &pad_attr.prepended : &pad_attr.appended;
  switch (axis) {
    case ::ml_drift::Axis::HEIGHT:
      p->h = zero_shape.h;
      break;
    case ::ml_drift::Axis::WIDTH:
      p->w = zero_shape.w;
      break;
    case ::ml_drift::Axis::CHANNELS:
      p->c = zero_shape.c;
      break;
    default:
      break;
  }
  pad_op->attr = pad_attr;

  const auto* params =
      static_cast<const TfLiteConcatenationParams*>(node.builtin_data);
  HandleFusedActivation(params->activation, ir_model, pad_op, tensor_map,
                        node.outputs->data[0]);
  return true;
}

}  // namespace

void ConvertConcat(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  if (TryConvertConcatToPad(context, node, tensor_map, ir_model)) {
    return;
  }

  std::vector<::ml_drift::ir::IrTensorId> input_ids;
  std::vector<::ml_drift::BHWDC> input_shapes;
  input_ids.reserve(node.inputs->size);
  input_shapes.reserve(node.inputs->size);
  for (int i = 0; i < node.inputs->size; ++i) {
    const TfLiteTensor* input_tensor = context.tensors + node.inputs->data[i];
    ::ml_drift::ir::IrTensorId input_id;
    if (tflite::IsConstantTensor(input_tensor)) {
      input_id = AddConstInput(context, node.inputs->data[i], ir_model, {})->id;
    } else {
      input_id = tensor_map[node.inputs->data[i]];
    }
    input_ids.push_back(input_id);
    input_shapes.push_back(
        ir_model.tensor(input_ids.back())->desc.GetBHWDCShape());
  }

  ::ml_drift::ir::IrOp* concat_op = ir_model.add_op();
  concat_op->name = ToString(::ml_drift::OperationType::CONCAT);

  for (int i = 0; i < input_ids.size(); ++i) {
    for (int j = 0; j < i; ++j) {
      if (input_ids[i] == input_ids[j]) {
        ::ml_drift::ir::IrOp* copy_op = ir_model.add_op();
        copy_op->name = ToString(::ml_drift::OperationType::COPY);
        ir_model.AddConsumer(input_ids[j], copy_op->id);
        const ::ml_drift::DataType dtype =
            ir_model.tensor(input_ids[j])->desc.GetDataType();
        const ::ml_drift::BHWDC shape =
            ir_model.tensor(input_ids[j])->desc.GetBHWDCShape();
        ::ml_drift::ir::IrTensor* copy_tensor =
            ir_model.add_tensor(dtype, shape);
        ir_model.SetProducer(copy_tensor->id, copy_op->id);
        input_ids[i] = copy_tensor->id;
        break;
      }
    }
  }

  for (::ml_drift::ir::IrTensorId input_id : input_ids) {
    ir_model.AddConsumer(input_id, concat_op->id);
  }

  ::ml_drift::ConcatAttributes attr;
  const ::ml_drift::BHWDC output_shape =
      ir_model.tensor(tensor_map[node.outputs->data[0]])->desc.GetBHWDCShape();
  attr.axis = GetConcatAxis(input_shapes, output_shape);
  concat_op->attr = attr;
  const auto* params =
      static_cast<const TfLiteConcatenationParams*>(node.builtin_data);
  HandleFusedActivation(params->activation, ir_model, concat_op, tensor_map,
                        node.outputs->data[0]);
}

}  // namespace litert::ml_drift::ir
