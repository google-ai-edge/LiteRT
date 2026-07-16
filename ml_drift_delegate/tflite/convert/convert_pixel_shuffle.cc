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

#include "ml_drift_delegate/tflite/convert/convert_pixel_shuffle.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertPixelShuffle(const TfLiteContext& context, const TfLiteNode& node,
                         const TfLiteRegistration& registration,
                         ::ml_drift::ir::TensorMap& tensor_map,
                         ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
  ir_op->name = ToString(::ml_drift::OperationType::DEPTH_TO_SPACE);

  const int input_id = tensor_map[node.inputs->data[0]];
  ir_model.AddConsumer(input_id, ir_op->id);

  ::ml_drift::SpaceToDepthAttributes attr;

  const uint8_t* buffer_t =
      reinterpret_cast<const uint8_t*>(node.custom_initial_data);
  size_t length = node.custom_initial_data_size;
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  attr.block_size = m["num_groups"].AsInt32();

  ir_op->attr = std::move(attr);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], ir_op->id);
}

}  // namespace litert::ml_drift::ir
