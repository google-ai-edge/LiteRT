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

#include "ml_drift_delegate/tflite/convert/convert_dequantize.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertDequantize(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* input_tensor = context.tensors + node.inputs->data[0];

  if (tflite::IsConstantTensor(input_tensor)) {
    ::ml_drift::ir::IrOp* op = ir_model.add_op();
    op->name = ToString(::ml_drift::OperationType::kConstant);

    ::ml_drift::ConstTensorAttributes attr;
    ::ml_drift::TensorFloat32 const_tensor;
    PopulateTensor(input_tensor, node.inputs->data[0], &const_tensor,
                   PopulateTensorFlags::kNoExtraBytes);

    attr.tensor = std::move(const_tensor);
    op->attr = std::move(attr);

    ir_model.SetProducer(tensor_map[node.outputs->data[0]], op->id);
    return;
  }

  const ::ml_drift::ir::IrTensorId input_ir_id =
      tensor_map[node.inputs->data[0]];
  const ::ml_drift::ir::IrTensorId output_ir_id =
      tensor_map[node.outputs->data[0]];

  if (input_tensor->type == kTfLiteFloat16) {
    ::ml_drift::ir::IrOp* op = ir_model.add_op();
    op->name = ToString(::ml_drift::OperationType::kCast);
    ir_model.AddConsumer(input_ir_id, op->id);
    ir_model.SetProducer(output_ir_id, op->id);
    return;
  }

  const ::ml_drift::DataType input_dtype =
      ir_model.tensor(input_ir_id)->desc.GetDataType();
  const ::ml_drift::DataType output_dtype =
      ir_model.tensor(output_ir_id)->desc.GetDataType();
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::kQuantizeAndDequantize);

  ir_model.AddConsumer(input_ir_id, op->id);

  ::ml_drift::ir::IrQuantParams params;
  PopulateQuantParams(*input_tensor, &params);

  ::ml_drift::QuantizeAndDequantizeAttributes attr;
  attr.min = params.min;
  attr.max = params.max;
  attr.scale = params.scale;
  op->attr = std::move(attr);

  if (output_dtype == input_dtype) {
    ir_model.SetProducer(output_ir_id, op->id);
    return;
  }

  // The output (e.g. FLOAT16) has another type than the float version of the
  // quantized input: QuantizeAndDequantize keeps the input type and a Cast
  // produces the output.
  const ::ml_drift::BHWDC output_shape =
      ir_model.tensor(output_ir_id)->desc.GetBHWDCShape();
  const ::ml_drift::ir::IrTensor* dequantized =
      ir_model.add_tensor(input_dtype, output_shape);
  ir_model.SetProducer(dequantized->id, op->id);
  ::ml_drift::ir::IrOp* cast_op = ir_model.add_op();
  cast_op->name = ToString(::ml_drift::OperationType::kCast);
  ir_model.AddConsumer(dequantized->id, cast_op->id);
  ir_model.SetProducer(output_ir_id, cast_op->id);
}

}  // namespace litert::ml_drift::ir
