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

#include "ml_drift_delegate/tflite/convert/convert_quantize.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertQuantize(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const ::ml_drift::ir::IrTensorId raw_input_id =
      tensor_map[node.inputs->data[0]];
  const ::ml_drift::ir::IrTensorId output_id =
      tensor_map[node.outputs->data[0]];
  const ::ml_drift::DataType input_dtype =
      ir_model.tensor(raw_input_id)->desc.GetDataType();
  const ::ml_drift::DataType output_dtype =
      ir_model.tensor(output_id)->desc.GetDataType();
  const ::ml_drift::BHWDC input_shape =
      ir_model.tensor(raw_input_id)->desc.GetBHWDCShape();

  ::ml_drift::ir::IrTensorId qdq_input_id = raw_input_id;
  if (input_dtype != output_dtype) {
    // QUANTIZE is emulated with QuantizeAndDequantize, which keeps the type of
    // its input. Cast the input (e.g. FLOAT16) to the float version of the
    // quantized output first.
    const ::ml_drift::ir::IrTensor* casted =
        ir_model.add_tensor(output_dtype, input_shape);
    ::ml_drift::ir::IrOp* cast_op = ir_model.add_op();
    cast_op->name = ToString(::ml_drift::OperationType::kCast);
    ir_model.AddConsumer(raw_input_id, cast_op->id);
    ir_model.SetProducer(casted->id, cast_op->id);
    qdq_input_id = casted->id;
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::kQuantizeAndDequantize);

  ir_model.AddConsumer(qdq_input_id, op->id);

  const TfLiteTensor* output_tensor = context.tensors + node.outputs->data[0];

  ::ml_drift::ir::IrQuantParams params;
  PopulateQuantParams(*output_tensor, &params);

  ::ml_drift::QuantizeAndDequantizeAttributes attr;
  attr.min = params.min;
  attr.max = params.max;
  attr.scale = params.scale;
  op->attr = std::move(attr);

  ir_model.SetProducer(output_id, op->id);
}

}  // namespace litert::ml_drift::ir
