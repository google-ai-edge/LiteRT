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

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertArgMax(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int dim_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];

  const TfLiteTensor* src_tensor = context.tensors + input_id;
  const TfLiteTensor* dim_tensor = context.tensors + dim_id;
  const TfLiteTensor* dst_tensor = context.tensors + output_id;

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::MAX_INDEX);

  ::ml_drift::MaxIndexAttributes attr;
  attr.dim = ExtractAxisFromIndex(*src_tensor, dim_tensor->data.i32[0]);

  ir_model.AddConsumer(tensor_map[input_id], op->id);

  if (src_tensor->dims->size != dst_tensor->dims->size) {
    ::ml_drift::BHWDC arg_max_shape = ExtractTensorShape(src_tensor->dims);
    arg_max_shape.set(attr.dim, 1);

    ::ml_drift::ir::IrTensor* arg_max_result = ir_model.add_tensor(
        ir_model.tensor(tensor_map[output_id])->desc.GetDataType(),
        arg_max_shape);

    ir_model.SetProducer(arg_max_result->id, op->id);

    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_attr;
    reshape_attr.new_shape =
        ir_model.tensor(tensor_map[output_id])->desc.GetBHWCShape();
    reshape_op->attr = std::move(reshape_attr);

    ir_model.AddConsumer(arg_max_result->id, reshape_op->id);
    ir_model.SetProducer(tensor_map[output_id], reshape_op->id);
  } else {
    ir_model.SetProducer(tensor_map[output_id], op->id);
  }

  op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
