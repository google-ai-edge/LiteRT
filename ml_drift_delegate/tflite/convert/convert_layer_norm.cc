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

#include "ml_drift_delegate/tflite/convert/convert_layer_norm.h"

#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertLayerNorm(const TfLiteContext& context, const TfLiteNode& node,
                      const TfLiteRegistration& registration,
                      ::ml_drift::ir::TensorMap& tensor_map,
                      ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* layer_norm_op = ir_model.add_op();
  layer_norm_op->name = ToString(::ml_drift::OperationType::LAYER_NORM);

  const int input_id = tensor_map[node.inputs->data[0]];
  ir_model.AddConsumer(input_id, layer_norm_op->id);

  ::ml_drift::LayerNormAttributes attr;

  if (node.inputs->size > 1) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> scale;
    PopulateTensor(&context.tensors[node.inputs->data[1]], node.inputs->data[1],
                   &scale, PopulateTensorFlags::kNoExtraBytes);
    attr.scale = std::move(scale);
  }
  if (node.inputs->size > 2) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> bias;
    PopulateTensor(&context.tensors[node.inputs->data[2]], node.inputs->data[2],
                   &bias, PopulateTensorFlags::kNoExtraBytes);
    attr.bias = std::move(bias);
  }

  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node.builtin_data);
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();

  attr.epsilon = flexbuffer_map["epsilon"].AsFloat();

  layer_norm_op->attr = std::move(attr);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], layer_norm_op->id);
}

}  // namespace litert::ml_drift::ir
