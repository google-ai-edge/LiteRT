// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kActiveTokensAlignedIndex = 2;

::ml_drift::Value* NewValueToMergeBatch(::ml_drift::GraphFloat32* graph,
                                        ::ml_drift::Value* val) {
  auto* new_val = graph->NewValue();
  new_val->tensor.type = val->tensor.type;
  new_val->tensor.shape =
      ::ml_drift::BHWC(1, val->tensor.shape.b * val->tensor.shape.h,
                       val->tensor.shape.w, val->tensor.shape.c);
  return new_val;
}

void Reshape(::ml_drift::GraphFloat32* graph, ::ml_drift::Value* before,
             ::ml_drift::Value* after) {
  ::ml_drift::Node* reshape = graph->NewNode();
  reshape->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = after->tensor.shape;
  reshape->operation.attributes = reshape_attr;
  graph->AddConsumer(reshape->id, before->id);
  graph->SetProducer(reshape->id, after->id);
}

::ml_drift::Value* ReshapeToMergeBatch(::ml_drift::GraphFloat32* graph,
                                       ::ml_drift::Value* val) {
  auto* new_val = NewValueToMergeBatch(graph, val);
  Reshape(graph, val, new_val);
  return new_val;
}

}  // namespace

absl::Status SdpaTransposedOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  int runtime_inputs = GetNumberOfRuntimeInputsForNode(context, tflite_node);
  if (runtime_inputs < 3 || runtime_inputs > 5) {
    return absl::UnavailableError(
        "SDPA transposed expects between 3 and 5 inputs.");
  }

  for (int i = 0; i < runtime_inputs; ++i) {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  return absl::OkStatus();
}

void SdpaTransposedOperationParser::Parse(const TfLiteNode* tflite_node,
                                          const TfLiteRegistration*,
                                          ::ml_drift::GraphFloat32* graph,
                                          ObjectReader* reader) {
  ::ml_drift::Value* query = reader->ReadValue(0);
  ::ml_drift::Value* key = reader->ReadValue(1);
  ::ml_drift::Value* value = reader->ReadValue(2);
  ::ml_drift::Value* mask = nullptr;
  ::ml_drift::Value* param_tensor = nullptr;
  if (tflite_node->inputs->size == 4) {
    auto* mask_or_param = reader->ReadValue(3);
    if (mask_or_param &&
        mask_or_param->tensor.type == ::ml_drift::DataType::INT32) {
      param_tensor = mask_or_param;
    } else {
      mask = mask_or_param;
    }
  } else if (tflite_node->inputs->size > 4) {
    mask = reader->ReadValue(3);
    param_tensor = reader->ReadValue(4);
  }

  ::ml_drift::Value* output =
      reader->ReadValueByTensorIdx(tflite_node->outputs->data[0]);

  const bool model_batch = query->tensor.shape.b != 1;
  ::ml_drift::Value* q_val = query;
  ::ml_drift::Value* k_val = key;
  ::ml_drift::Value* v_val = value;
  ::ml_drift::Value* result_val = output;
  // Reshape to merge batch dimension as the kernel expects 1xBxMxN shape.
  if (model_batch) {
    q_val = ReshapeToMergeBatch(graph, query);
    k_val = ReshapeToMergeBatch(graph, key);
    v_val = ReshapeToMergeBatch(graph, value);
    result_val = NewValueToMergeBatch(graph, output);
  }

  SdpaTransposedAttributes attr;
  if (param_tensor) {
    attr.runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
  }
  attr.is_prefill = (query->tensor.shape.w > 2 || model_batch);

  const ::ml_drift::BHWC right_shape_k = k_val->tensor.shape;
  attr.bmm1_weights.weights_shape =
      ::ml_drift::OHWI(right_shape_k.w, right_shape_k.h, 1, right_shape_k.c);
  // TODO: b/403337563 - Support quantized weights.
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      k_val->tensor.type, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

  const ::ml_drift::BHWC right_shape_v = v_val->tensor.shape;
  attr.bmm2_weights.weights_shape =
      ::ml_drift::OHWI(right_shape_v.w, right_shape_v.h, 1, right_shape_v.c);
  // TODO: byungchul - Support quantized weights.
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      v_val->tensor.type, attr.bmm2_weights.weights_shape);

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params && params->attributes) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["softcap"].IsNull()) {
      attr.softcap = flexbuffer_map["softcap"].AsFloat();
    }
  }

  ::ml_drift::Node* node = graph->NewNode();
  node->operation.attributes = std::move(attr);
  node->operation.type = kSdpaTransposedType;
  graph->AddConsumer(node->id, q_val->id);
  graph->AddConsumer(node->id, k_val->id);
  graph->AddConsumer(node->id, v_val->id);
  if (mask) {
    graph->AddConsumer(node->id, mask->id);
  }
  if (param_tensor) {
    graph->AddConsumer(node->id, param_tensor->id);
  }
  graph->SetProducer(node->id, result_val->id);

  if (model_batch) {
    Reshape(graph, result_val, output);
  }
}

}  // namespace litert::ml_drift
