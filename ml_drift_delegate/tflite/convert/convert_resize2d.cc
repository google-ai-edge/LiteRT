// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/convert/convert_resize2d.h"

#include <utility>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertResize2d(const TfLiteContext& context, const TfLiteNode& node,
                     const TfLiteRegistration& registration,
                     ::ml_drift::ir::TensorMap& tensor_map,
                     ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
  ir_op->name = ToString(::ml_drift::OperationType::RESIZE);

  const int input_id = node.inputs->data[0];
  ir_model.AddConsumer(tensor_map[input_id], ir_op->id);

  const int output_id = node.outputs->data[0];
  ir_model.SetProducer(tensor_map[output_id], ir_op->id);

  ::ml_drift::Resize2DAttributes attr;
  const ::ml_drift::BHWC output_shape =
      ir_model.tensor(tensor_map[output_id])->desc.GetBHWCShape();
  attr.new_shape = ::ml_drift::HW(output_shape.h, output_shape.w);

  if (registration.builtin_code == kTfLiteBuiltinResizeBilinear) {
    attr.type = ::ml_drift::SamplingType::BILINEAR;
    const auto* params =
        static_cast<const TfLiteResizeBilinearParams*>(node.builtin_data);
    if (params) {
      attr.align_corners = params->align_corners;
      attr.half_pixel_centers = params->half_pixel_centers;
    }
  } else if (registration.builtin_code == kTfLiteBuiltinResizeNearestNeighbor) {
    attr.type = ::ml_drift::SamplingType::NEAREST;
    const auto* params = static_cast<const TfLiteResizeNearestNeighborParams*>(
        node.builtin_data);
    if (params) {
      attr.align_corners = params->align_corners;
      attr.half_pixel_centers = params->half_pixel_centers;
    }
  }

  ir_op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
