// Copyright 2026 The ML Drift Authors.
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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_topk.h"

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertTopK(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int k_id = node.inputs->data[1];
  const int output_id_values = node.outputs->data[0];
  const int output_id_indices = node.outputs->data[1];
  const TfLiteTensor* input_tensor = context.tensors + input_id;
  const TfLiteTensor* k_tensor = context.tensors + k_id;

  const bool is_1d = input_tensor->dims->size == 1;
  ::ml_drift::ir::IrTensorId final_input_id = tensor_map[input_id];

  // Insert a RESHAPE if input tensor [N] is mis-auto-expanded to [N,1,1,1].
  if (is_1d) {
    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);

    ::ml_drift::ReshapeAttributes reshape_attr;
    ::ml_drift::BHWC new_shape(1, 1, 1, input_tensor->dims->data[0]);
    reshape_attr.new_shape = new_shape;
    reshape_op->attr = std::move(reshape_attr);

    ir_model.AddConsumer(tensor_map[input_id], reshape_op->id);

    ::ml_drift::ir::IrTensor* new_input_tensor = ir_model.add_tensor(
        ir_model.tensor(tensor_map[input_id])->desc.GetDataType(), new_shape);
    final_input_id = new_input_tensor->id;

    ir_model.SetProducer(final_input_id, reshape_op->id);
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::TOP_K);
  ::ml_drift::TopKAttributes attr;

  const int32_t k = k_tensor->data.i32[0];
  attr.k = k;
  op->attr = std::move(attr);
  ir_model.AddConsumer(final_input_id, op->id);

  // Reshape the output tensors back
  if (is_1d) {
    ::ml_drift::ir::IrTensor* top_k_interim_values = ir_model.add_tensor(
        ir_model.tensor(tensor_map[output_id_values])->desc.GetDataType(),
        ::ml_drift::BHWC(1, 1, 1, k));
    ::ml_drift::ir::IrTensor* top_k_interim_indices = ir_model.add_tensor(
        ir_model.tensor(tensor_map[output_id_indices])->desc.GetDataType(),
        ::ml_drift::BHWC(1, 1, 1, k));

    ir_model.SetProducer(top_k_interim_values->id, op->id);
    ir_model.SetProducer(top_k_interim_indices->id, op->id);

    ::ml_drift::ir::IrOp* reshape_values_op = ir_model.add_op();
    reshape_values_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_values_attr;
    reshape_values_attr.new_shape = ::ml_drift::BHWC(k, 1, 1, 1);
    reshape_values_op->attr = std::move(reshape_values_attr);
    ir_model.AddConsumer(top_k_interim_values->id, reshape_values_op->id);
    ir_model.SetProducer(tensor_map[output_id_values], reshape_values_op->id);

    ::ml_drift::ir::IrOp* reshape_indices_op = ir_model.add_op();
    reshape_indices_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_indices_attr;
    reshape_indices_attr.new_shape = ::ml_drift::BHWC(k, 1, 1, 1);
    reshape_indices_op->attr = std::move(reshape_indices_attr);
    ir_model.AddConsumer(top_k_interim_indices->id, reshape_indices_op->id);
    ir_model.SetProducer(tensor_map[output_id_indices], reshape_indices_op->id);
  } else {
    ir_model.SetProducer(tensor_map[output_id_values], op->id);
    ir_model.SetProducer(tensor_map[output_id_indices], op->id);
  }
}

}  // namespace litert::ml_drift::ir
