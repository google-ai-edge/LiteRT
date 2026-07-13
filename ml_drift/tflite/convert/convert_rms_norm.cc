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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_rms_norm.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertRmsNorm(const TfLiteContext& context, const TfLiteNode& node,
                    const TfLiteRegistration& registration,
                    ::ml_drift::ir::TensorMap& tensor_map,
                    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* rms_norm_op = ir_model.add_op();
  rms_norm_op->name = ToString(::ml_drift::OperationType::RMS_NORM);

  const int input_id = tensor_map[node.inputs->data[0]];
  ir_model.AddConsumer(input_id, rms_norm_op->id);

  ::ml_drift::RmsNormAttributes attr;

  if (node.inputs->size > 1) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> scale;
    PopulateTensor(&context.tensors[node.inputs->data[1]], node.inputs->data[1],
                   &scale, PopulateTensorFlags::kNoExtraBytes);
    attr.scale = std::move(scale);
  }

  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node.builtin_data);
  const uint8_t* buffer_t =
      reinterpret_cast<const uint8_t*>(params->attributes);
  size_t length = params->attributes_size;

  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(buffer_t, length).AsMap();

  attr.epsilon = flexbuffer_map["epsilon"].AsFloat();

  rms_norm_op->attr = std::move(attr);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], rms_norm_op->id);
}

}  // namespace litert::ml_drift::ir
