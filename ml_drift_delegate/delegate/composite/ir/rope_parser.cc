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

#include "ml_drift_delegate/delegate/composite/ir/rope_parser.h"

#include <cstddef>
#include <cstdint>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

// RoPE is applied either to a single tensor (input, positions) or to the query
// and key tensors together (query, key, positions), producing one output per
// rotated tensor.
absl::Status RopeIsSupported(const TfLiteContext* /*context*/,
                             const TfLiteNode* tflite_node,
                             const TfLiteRegistration* /*registration*/) {
  if (tflite_node->inputs->size == 2) {
    if (tflite_node->outputs->size != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("RoPE with 2 inputs expects 1 output, but got ",
                       tflite_node->outputs->size));
    }
  } else if (tflite_node->inputs->size == 3) {
    if (tflite_node->outputs->size != 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("RoPE with 3 inputs expects 2 outputs, but got ",
                       tflite_node->outputs->size));
    }
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "RoPE expects 2 or 3 inputs, but got ", tflite_node->inputs->size));
  }
  return absl::OkStatus();
}

// Reads the RoPE attributes out of a flexbuffer. The exporters have used
// several spellings for the timescale and partial-rotary settings, so the
// aliases accepted by the non-IR parser are accepted here too.
void ParseRopeAttributes(const uint8_t* buffer, size_t length,
                         ::ml_drift::RoPEAttributes& attr) {
  if (buffer == nullptr || length == 0) return;
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(buffer, length).AsMap();

  if (!flexbuffer_map["min_timescale"].IsNull()) {
    attr.min_timescale = flexbuffer_map["min_timescale"].AsFloat();
  }
  if (!flexbuffer_map["max_timescale"].IsNull()) {
    attr.max_timescale = flexbuffer_map["max_timescale"].AsFloat();
  } else if (!flexbuffer_map["base"].IsNull()) {
    attr.max_timescale = flexbuffer_map["base"].AsFloat();
  } else if (!flexbuffer_map["rope_theta"].IsNull()) {
    attr.max_timescale = flexbuffer_map["rope_theta"].AsFloat();
  } else if (!flexbuffer_map["theta"].IsNull()) {
    attr.max_timescale = flexbuffer_map["theta"].AsFloat();
  }
  if (!flexbuffer_map["proportion"].IsNull()) {
    attr.proportion = flexbuffer_map["proportion"].AsFloat();
  } else if (!flexbuffer_map["partial_rotary_factor"].IsNull()) {
    attr.proportion = flexbuffer_map["partial_rotary_factor"].AsFloat();
  }
  if (!flexbuffer_map["kernel_type"].IsNull()) {
    attr.kernel_type = static_cast<::ml_drift::RoPEKernelType>(
        flexbuffer_map["kernel_type"].AsInt32());
  }
}

void RopeConvert(
    const TfLiteContext& /*context*/, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  // RoPE is a native ML Drift operation rather than a custom delegate kernel,
  // so the op is named after `OperationType::ROPE` and is picked up by the
  // stock operation selector instead of LiteRtOpSelector.
  op->name = "rope";

  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    if (tflite_node.inputs->data[i] != kTfLiteOptionalTensor) {
      ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[i]], op->id);
    }
  }
  for (int i = 0; i < tflite_node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[tflite_node.outputs->data[i]], op->id);
  }

  ::ml_drift::RoPEAttributes attr;
  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  if (params != nullptr && params->attributes != nullptr &&
      params->attributes_size > 0) {
    ParseRopeAttributes(reinterpret_cast<const uint8_t*>(params->attributes),
                        params->attributes_size, attr);
  } else if (tflite_node.custom_initial_data != nullptr &&
             tflite_node.custom_initial_data_size > 0) {
    ParseRopeAttributes(
        reinterpret_cast<const uint8_t*>(tflite_node.custom_initial_data),
        tflite_node.custom_initial_data_size, attr);
  }
  op->attr = attr;
}

}  // namespace

CustomIrOpParser GetRopeParser() { return {RopeIsSupported, RopeConvert}; }

}  // namespace litert::ml_drift::ir
