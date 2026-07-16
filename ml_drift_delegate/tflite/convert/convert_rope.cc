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

#include "ml_drift_delegate/tflite/convert/convert_rope.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertRoPE(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::ROPE);

  // Either 2 inputs -> 1 output, or 3 inputs -> 2 outputs.
  for (int i = 0; i < node.inputs->size; ++i) {
    ir_model.AddConsumer(tensor_map[node.inputs->data[i]], op->id);
  }

  for (int i = 0; i < node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[node.outputs->data[i]], op->id);
  }

  ::ml_drift::RoPEAttributes attr;
  const uint8_t* buffer_t = nullptr;
  size_t length = 0;
  if (node.custom_initial_data && node.custom_initial_data_size > 0) {
    buffer_t = reinterpret_cast<const uint8_t*>(node.custom_initial_data);
    length = node.custom_initial_data_size;
  } else if (node.builtin_data) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node.builtin_data);
    if (composite_params && composite_params->attributes &&
        composite_params->attributes_size > 0) {
      buffer_t = reinterpret_cast<const uint8_t*>(composite_params->attributes);
      length = composite_params->attributes_size;
    }
  }
  if (buffer_t && length > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(buffer_t, length).AsMap();
    if (!flexbuffer_map["min_timescale"].IsNull()) {
      attr.min_timescale = flexbuffer_map["min_timescale"].AsFloat();
    }
    if (!flexbuffer_map["max_timescale"].IsNull()) {
      attr.max_timescale = flexbuffer_map["max_timescale"].AsFloat();
    }
    if (!flexbuffer_map["proportion"].IsNull()) {
      attr.proportion = flexbuffer_map["proportion"].AsFloat();
    }
    if (!flexbuffer_map["interleaved"].IsNull()) {
      attr.interleaved = flexbuffer_map["interleaved"].AsBool();
    }
    if (!flexbuffer_map["axial_dims"].IsNull()) {
      attr.axial_dims = flexbuffer_map["axial_dims"].AsInt32();
    }
  }
  op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
