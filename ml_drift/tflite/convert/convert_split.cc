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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_split.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

namespace {

void ConvertSplitCommon(
    const TfLiteContext& context, const TfLiteNode& node, int input_idx,
    int axis_idx, int num_splits,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  if (num_splits == 1) {
    // Adding Identity reshape that will be removed.
    ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
    ir_op->name = ToString(::ml_drift::OperationType::RESHAPE);

    const int input_id = node.inputs->data[input_idx];
    ir_model.AddConsumer(tensor_map.at(input_id), ir_op->id);

    const int output_id = node.outputs->data[0];
    ir_model.SetProducer(tensor_map.at(output_id), ir_op->id);

    ::ml_drift::ReshapeAttributes attr;
    attr.new_shape =
        ir_model.tensor(tensor_map.at(output_id))->desc.GetBHWCShape();
    ir_op->attr = std::move(attr);
    return;
  }

  const TfLiteTensor* input_tensor =
      context.tensors + node.inputs->data[input_idx];
  const TfLiteTensor* axis_tensor =
      context.tensors + node.inputs->data[axis_idx];

  ::ml_drift::SplitAttributes attr;
  attr.axis = ExtractAxisFromIndex(*input_tensor, axis_tensor->data.i32[0]);

  ::ml_drift::ir::IrOp* split_op = ir_model.add_op();
  split_op->name = ToString(::ml_drift::OperationType::SPLIT);
  split_op->attr = attr;

  const int input_id = node.inputs->data[input_idx];
  ir_model.AddConsumer(tensor_map.at(input_id), split_op->id);

  for (int i = 0; i < node.outputs->size; ++i) {
    const int output_id = node.outputs->data[i];
    ir_model.SetProducer(tensor_map.at(output_id), split_op->id);
  }
}

}  // namespace

void ConvertSplit(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const auto* params = static_cast<const TfLiteSplitParams*>(node.builtin_data);
  ConvertSplitCommon(context, node, /*input_idx=*/1, /*axis_idx=*/0,
                     params->num_splits, tensor_map, ir_model);
}

void ConvertSplitV(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const auto* params =
      static_cast<const TfLiteSplitVParams*>(node.builtin_data);
  ConvertSplitCommon(context, node, /*input_idx=*/0, /*axis_idx=*/2,
                     params->num_splits, tensor_map, ir_model);
}

}  // namespace litert::ml_drift::ir
