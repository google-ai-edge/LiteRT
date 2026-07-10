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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_unpooling2d.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertUnpooling2d(const TfLiteContext& context, const TfLiteNode& node,
                        const TfLiteRegistration& registration,
                        ::ml_drift::ir::TensorMap& tensor_map,
                        ::ml_drift::ir::IrModel& ir_model) {
  const auto* params =
      static_cast<const TfLitePoolParams*>(node.custom_initial_data);

  ::ml_drift::MaxUnpooling2DAttributes attr;
  attr.kernel = ToHW(params->filter_height, params->filter_width);
  attr.strides = ToHW(params->stride_height, params->stride_width);

  const int input_id = node.inputs->data[0];
  const ::ml_drift::ir::IrTensor* input_tensor =
      ir_model.tensor(tensor_map[input_id]);
  UpdatePadding(params->padding, input_tensor->desc.GetBHWDCShape(), &attr);

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::MAX_UNPOOLING_2D);

  // The first input is the tensor to be unpooled
  ir_model.AddConsumer(tensor_map[input_id], op->id);
  // The second input is the indices
  ir_model.AddConsumer(tensor_map[node.inputs->data[1]], op->id);

  const int output_id = node.outputs->data[0];
  HandleFusedActivation(params->activation, ir_model, op, tensor_map,
                        output_id);

  op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
