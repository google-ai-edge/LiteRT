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

#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/add_values_to_cache_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

namespace {

absl::Status AddValuesToCacheIsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* /*registration*/) {
  int num_runtime_inputs = 0;
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    if (tflite_node->inputs->data[i] != kTfLiteOptionalTensor &&
        !::tflite::IsConstantTensor(
            &context->tensors[tflite_node->inputs->data[i]])) {
      num_runtime_inputs++;
    }
  }

  if (num_runtime_inputs != 3 && num_runtime_inputs != 7) {
    return absl::UnavailableError(
        "odml.cache_update expects 3 or 7 runtime inputs.");
  }

  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError("odml.cache_update expects 2 outputs.");
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (flexbuffer_map["kv_cache_batch_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing kv_cache_batch_size.");
    }
    if (flexbuffer_map["cache_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing cache_size.");
    }
    if (flexbuffer_map["head_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing head_size.");
    }
  }

  return absl::OkStatus();
}

void AddValuesToCacheConvert(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* add_values_op = ir_model.add_op();
  add_values_op->name = "add_values_to_cache";

  ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[0]],
                       add_values_op->id);  // src_k
  ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[1]],
                       add_values_op->id);  // src_v
  ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[2]],
                       add_values_op->id);  // runtime_param_tensor

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
  ::litert::ml_drift::AddValuesToCacheAttributes attr;
  attr.kv_cache_batch_size = flexbuffer_map["kv_cache_batch_size"].AsInt32();
  attr.cache_size = flexbuffer_map["cache_size"].AsInt32();
  attr.head_size = flexbuffer_map["head_size"].AsInt32();
  if (!flexbuffer_map["scale_k"].IsNull()) {
    attr.scale_k = flexbuffer_map["scale_k"].AsFloat();
  }
  if (!flexbuffer_map["scale_v"].IsNull()) {
    attr.scale_v = flexbuffer_map["scale_v"].AsFloat();
  }
  add_values_op->attr = std::move(attr);

  ir_model.SetProducer(tensor_map[tflite_node.outputs->data[0]],
                       add_values_op->id);
  ir_model.SetProducer(tensor_map[tflite_node.outputs->data[1]],
                       add_values_op->id);
}

}  // namespace

CustomIrOpParser GetAddValuesToCacheParser() {
  return {
      .is_supported = AddValuesToCacheIsSupported,
      .convert = AddValuesToCacheConvert,
  };
}

}  // namespace litert::ml_drift::ir
