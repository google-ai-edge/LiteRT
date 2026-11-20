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

#include "ml_drift_delegate/tflite/convert/convert_scatter_nd.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"      // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertScatterNd(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  // The updates may be a constant, which needs a CONSTANT op to produce it,
  // unless it shares a buffer (or is external), in which case it already has
  // an IR tensor. Create the op first so that it precedes the scatter_nd op.
  const int updates_id = node.inputs->data[1];
  const auto updates_it = tensor_map.find(updates_id);
  const bool needs_const_op =
      ::tflite::IsConstantTensor(context.tensors + updates_id) &&
      (updates_it == tensor_map.end() ||
       !ir_model.tensor(updates_it->second)->buffer_source.is_shared);
  const ::ml_drift::ir::IrTensorId updates_tensor =
      needs_const_op ? AddConstInput(context, updates_id, ir_model, {})->id
                     : tensor_map.at(updates_id);

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = "scatter_nd";

  ir_model.AddConsumer(tensor_map.at(node.inputs->data[0]), op->id);
  ir_model.AddConsumer(updates_tensor, op->id);
  ir_model.SetProducer(tensor_map.at(node.outputs->data[0]), op->id);
}

}  // namespace litert::ml_drift::ir
