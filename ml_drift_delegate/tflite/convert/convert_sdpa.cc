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

#include "ml_drift_delegate/tflite/convert/convert_sdpa.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertSdpa(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* sdpa_op = ir_model.add_op();
  sdpa_op->name =
      ToString(::ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION);

  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], sdpa_op->id);  // Q
  ir_model.AddConsumer(tensor_map[node.inputs->data[1]], sdpa_op->id);  // K
  ir_model.AddConsumer(tensor_map[node.inputs->data[2]], sdpa_op->id);  // V

  if (node.inputs->size > 3 && node.inputs->data[3] != kTfLiteOptionalTensor) {
    const int mask_id = node.inputs->data[3];
    if (tflite::IsConstantTensor(context.tensors + mask_id)) {
      ::ml_drift::ir::IrTensor* const_tensor =
          AddConstInput(context, mask_id, ir_model, {});
      ir_model.AddConsumer(const_tensor->id, sdpa_op->id);
    } else {
      ir_model.AddConsumer(tensor_map[mask_id], sdpa_op->id);
    }
  }

  ir_model.SetProducer(tensor_map[node.outputs->data[0]], sdpa_op->id);

  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node.builtin_data);
  ::ml_drift::ScaledDotProductAttentionAttributes attr;
  if (params && params->attributes) {
    const flexbuffers::Map m =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (!m["scale"].IsNull()) {
      attr.scale = m["scale"].AsFloat();
    }
  }
  sdpa_op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
