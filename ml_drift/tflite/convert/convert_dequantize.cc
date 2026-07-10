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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_dequantize.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
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
    op->name = ToString(::ml_drift::OperationType::CONSTANT);

    ::ml_drift::ConstTensorAttributes attr;
    ::ml_drift::TensorFloat32 const_tensor;
    PopulateTensor(input_tensor, node.inputs->data[0], &const_tensor,
                   PopulateTensorFlags::kNoExtraBytes);

    attr.tensor = std::move(const_tensor);
    op->attr = std::move(attr);

    ir_model.SetProducer(tensor_map[node.outputs->data[0]], op->id);
    return;
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::QUANTIZE_AND_DEQUANTIZE);

  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], op->id);

  ::ml_drift::ir::IrQuantParams params;
  PopulateQuantParams(*input_tensor, &params);

  ::ml_drift::QuantizeAndDequantizeAttributes attr;
  attr.min = params.min;
  attr.max = params.max;
  attr.scale = params.scale;
  op->attr = std::move(attr);

  ir_model.SetProducer(tensor_map[node.outputs->data[0]], op->id);
}

}  // namespace litert::ml_drift::ir
