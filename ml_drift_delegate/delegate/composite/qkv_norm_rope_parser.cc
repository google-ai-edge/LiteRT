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

#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

absl::Status QkvNormRopeOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  const int num_inputs = tflite_node->inputs->size;
  const int num_outputs = tflite_node->outputs->size;
  const bool is_full_qkv = (num_inputs == 4 && num_outputs == 3);
  const bool is_q_only = (num_inputs == 3 && num_outputs == 1);
  if (!is_full_qkv && !is_q_only) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid inputs/outputs for QkvNormRoPE: got ", num_inputs,
        " inputs and ", num_outputs,
        " outputs, expected (4, 3) for QKV or (3, 1) for Q-only."));
  }

  for (int i = 0; i < num_inputs; ++i) {
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  return absl::OkStatus();
}

void QkvNormRopeOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    ::ml_drift::GraphFloat32* graph, ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kQkvNormRopeType;

  QkvNormRopeAttributes attr;
  const auto* params = reinterpret_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params && params->attributes && params->attributes_size > 0) {
    const flexbuffers::Map map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(params->attributes),
            params->attributes_size)
            .AsMap();
    if (!map["num_heads"].IsNull()) {
      attr.num_heads = map["num_heads"].AsInt32();
    }
    if (!map["num_kv_heads"].IsNull()) {
      attr.num_kv_heads = map["num_kv_heads"].AsInt32();
    }
    if (!map["head_dim"].IsNull()) {
      attr.head_dim = map["head_dim"].AsInt32();
    }
    if (!map["min_timescale"].IsNull()) {
      attr.min_timescale = map["min_timescale"].AsFloat();
    }
    if (!map["max_timescale"].IsNull()) {
      attr.max_timescale = map["max_timescale"].AsFloat();
    }
    if (!map["proportion"].IsNull()) {
      attr.proportion = map["proportion"].AsFloat();
    }
    if (!map["epsilon"].IsNull()) {
      attr.epsilon = map["epsilon"].AsFloat();
    }
    if (!map["has_v_norm"].IsNull()) {
      attr.has_v_norm = map["has_v_norm"].AsBool();
    }
  } else if (tflite_node->custom_initial_data &&
             tflite_node->custom_initial_data_size > 0) {
    auto root = flexbuffers::GetRoot(
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data),
        tflite_node->custom_initial_data_size);
    if (root.IsMap()) {
      auto map = root.AsMap();
      if (!map["num_heads"].IsNull()) {
        attr.num_heads = map["num_heads"].AsInt32();
      }
      if (!map["num_kv_heads"].IsNull()) {
        attr.num_kv_heads = map["num_kv_heads"].AsInt32();
      }
      if (!map["head_dim"].IsNull()) {
        attr.head_dim = map["head_dim"].AsInt32();
      }
      if (!map["min_timescale"].IsNull()) {
        attr.min_timescale = map["min_timescale"].AsFloat();
      }
      if (!map["max_timescale"].IsNull()) {
        attr.max_timescale = map["max_timescale"].AsFloat();
      }
      if (!map["proportion"].IsNull()) {
        attr.proportion = map["proportion"].AsFloat();
      }
      if (!map["epsilon"].IsNull()) {
        attr.epsilon = map["epsilon"].AsFloat();
      }
      if (!map["has_v_norm"].IsNull()) {
        attr.has_v_norm = map["has_v_norm"].AsBool();
      }
    }
  }
  if (tflite_node->inputs->size == 3) {
    attr.num_kv_heads = 0;
  }
  node->operation.attributes = attr;

  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    if (reader->IsConstantTensor(i)) {
      const ::ml_drift::Value* input = reader->AddConstInput(i, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    } else {
      reader->AddInput(node, i);
    }
  }

  reader->AddOutputs(node);
}

}  // namespace litert::ml_drift
