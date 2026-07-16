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

#include "ml_drift_delegate/tflite/convert/convert_bitcast.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertBitcast(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* src_tensor = context.tensors + node.inputs->data[0];
  const TfLiteTensor* dst_tensor = context.tensors + node.outputs->data[0];

  const int input_id = node.inputs->data[0];
  const int output_id = node.outputs->data[0];

  if (src_tensor->dims->size >
      dst_tensor->dims->size) {  // decrease precision size
    // bitcast -> reshape.
    // Ex: for si8 -> si32 we might have shapes such as
    //     (2, 2, 4) -> (bitcast) -> (2, 2, 1) -> (reshape) -> (2, 2)
    ::ml_drift::ir::IrOp* bitcast_op = ir_model.add_op();
    bitcast_op->name = ToString(::ml_drift::OperationType::BITCAST);
    ir_model.AddConsumer(tensor_map[input_id], bitcast_op->id);

    ::ml_drift::BHWDC interim_shape = ExtractTensorShape(src_tensor->dims);
    interim_shape.c = 1;

    const ::ml_drift::BHWDC output_shape = ExtractTensorShape(dst_tensor->dims);

    if (output_shape != interim_shape) {  // check reshape needed
      ::ml_drift::ir::IrTensor* interim_val = ir_model.add_tensor(
          ir_model.tensor(tensor_map[output_id])->desc.GetDataType(),
          interim_shape);
      ir_model.SetProducer(interim_val->id, bitcast_op->id);

      ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
      reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape =
          ir_model.tensor(tensor_map[output_id])->desc.GetBHWCShape();
      reshape_op->attr = std::move(reshape_attr);

      ir_model.AddConsumer(interim_val->id, reshape_op->id);
      ir_model.SetProducer(tensor_map[output_id], reshape_op->id);
    } else {
      ir_model.SetProducer(tensor_map[output_id], bitcast_op->id);
    }
  } else if (src_tensor->dims->size <
             dst_tensor->dims->size) {  // increase precision size
    // reshape -> bitcast
    // Ex: for si32 -> si8 we might have shapes such as
    //     (2, 2) -> (reshape) -> (2, 2, 1) -> (bitcast) -> (2, 2, 4)
    ::ml_drift::BHWDC interim_shape = ExtractTensorShape(dst_tensor->dims);
    interim_shape.c = 1;

    const ::ml_drift::BHWDC input_shape = ExtractTensorShape(src_tensor->dims);

    if (input_shape != interim_shape) {  // check reshape needed
      ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
      reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
      ir_model.AddConsumer(tensor_map[input_id], reshape_op->id);

      ::ml_drift::ir::IrTensor* interim_val = ir_model.add_tensor(
          ir_model.tensor(tensor_map[input_id])->desc.GetDataType(),
          interim_shape);

      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape = ::ml_drift::BHWC(
          interim_shape.b, interim_shape.h, interim_shape.w, interim_shape.c);
      reshape_op->attr = std::move(reshape_attr);
      ir_model.SetProducer(interim_val->id, reshape_op->id);

      ::ml_drift::ir::IrOp* bitcast_op = ir_model.add_op();
      bitcast_op->name = ToString(::ml_drift::OperationType::BITCAST);
      ir_model.AddConsumer(interim_val->id, bitcast_op->id);
      ir_model.SetProducer(tensor_map[output_id], bitcast_op->id);
    } else {
      ::ml_drift::ir::IrOp* bitcast_op = ir_model.add_op();
      bitcast_op->name = ToString(::ml_drift::OperationType::BITCAST);
      ir_model.AddConsumer(tensor_map[input_id], bitcast_op->id);
      ir_model.SetProducer(tensor_map[output_id], bitcast_op->id);
    }
  } else {  // maintain precision size
    // bitcast
    // Ex: for f32 -> si32 we might have shapes such as
    //     (2, 2, 4) -> (bitcast) -> (2, 2, 4)
    ::ml_drift::ir::IrOp* bitcast_op = ir_model.add_op();
    bitcast_op->name = ToString(::ml_drift::OperationType::BITCAST);
    ir_model.AddConsumer(tensor_map[input_id], bitcast_op->id);
    ir_model.SetProducer(tensor_map[output_id], bitcast_op->id);
  }
}

}  // namespace litert::ml_drift::ir
