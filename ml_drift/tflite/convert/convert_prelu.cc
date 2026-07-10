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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_prelu.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertPrelu(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::PReLUAttributes attr;

  const TfLiteTensor* alpha_tensor = context.tensors + node.inputs->data[1];

  if (alpha_tensor->dims->size == 1 ||
      (alpha_tensor->dims->size == 3 && alpha_tensor->dims->data[0] == 1 &&
       alpha_tensor->dims->data[1] == 1)) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> alpha;
    PopulateTensor(alpha_tensor, 0, &alpha, PopulateTensorFlags::kNoExtraBytes);
    attr.alpha = std::move(alpha);
  } else if (alpha_tensor->dims->size == 3 || alpha_tensor->dims->size == 4) {
    ::ml_drift::Tensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32> alpha;
    PopulateTensor(alpha_tensor, 0, &alpha, PopulateTensorFlags::kNoExtraBytes);
    attr.alpha = std::move(alpha);
  } else {
    ABSL_LOG(FATAL) << "Unsupported PRelu alpha shape";
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::PRELU);
  op->attr = std::move(attr);
  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], op->id);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], op->id);
}

}  // namespace litert::ml_drift::ir
