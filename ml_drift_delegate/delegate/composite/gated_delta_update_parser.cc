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

#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

GatedDeltaUpdateAttributes ParseGatedDeltaUpdateAttributes(
    const TfLiteNode& tflite_node, bool block_form) {
  GatedDeltaUpdateAttributes attr;
  if (block_form) {
    attr.block = GatedDeltaNetAttributes();
  }
  const uint8_t* buffer_t = nullptr;
  size_t length = 0;
  if (tflite_node.custom_initial_data &&
      tflite_node.custom_initial_data_size > 0) {
    buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node.custom_initial_data);
    length = tflite_node.custom_initial_data_size;
  } else if (tflite_node.builtin_data) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node.builtin_data);
    if (composite_params && composite_params->attributes &&
        composite_params->attributes_size > 0) {
      buffer_t = reinterpret_cast<const uint8_t*>(composite_params->attributes);
      length = composite_params->attributes_size;
    }
  }
  if (buffer_t && length > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(buffer_t, length).AsMap();
    if (!flexbuffer_map["mode"].IsNull()) {
      attr.mode = flexbuffer_map["mode"].AsInt32();
    }
    if (!flexbuffer_map["state_dtype"].IsNull()) {
      attr.state_dtype = flexbuffer_map["state_dtype"].AsString().str();
    }
    if (attr.block.has_value()) {
      if (!flexbuffer_map["eps"].IsNull()) {
        attr.block->eps = flexbuffer_map["eps"].AsFloat();
      }
      if (!flexbuffer_map["norm_eps"].IsNull()) {
        attr.block->norm_eps = flexbuffer_map["norm_eps"].AsFloat();
      }
      if (!flexbuffer_map["output_gate_type"].IsNull()) {
        attr.block->output_gate_type =
            flexbuffer_map["output_gate_type"].AsString().str();
      } else if (!flexbuffer_map["gate_activation"].IsNull()) {
        attr.block->output_gate_type =
            flexbuffer_map["gate_activation"].AsString().str();
      }
    }
  }
  return attr;
}

absl::Status SetGatedDeltaNetDims(const TfLiteIntArray& conv_out_dims,
                                  const TfLiteIntArray& state_dims,
                                  GatedDeltaNetAttributes* attr) {
  if (conv_out_dims.size != 3 || state_dims.size != 4) {
    return absl::UnavailableError(
        "gated_delta_update block mode expects conv_out [1, T, conv_dim] and "
        "recurrent_state [1, H_v, D_k, D_v].");
  }
  const int conv_dim = conv_out_dims.data[2];
  const int num_v_heads = state_dims.data[1];
  const int head_k_dim = state_dims.data[2];
  const int head_v_dim = state_dims.data[3];
  if (num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0) {
    return absl::UnavailableError(
        "gated_delta_update block mode expects positive H_v, D_k and D_v in "
        "recurrent_state [1, H_v, D_k, D_v].");
  }
  const int value_dim = num_v_heads * head_v_dim;
  if (conv_dim <= value_dim || (conv_dim - value_dim) % 2 != 0) {
    return absl::UnavailableError(
        "gated_delta_update block mode expects conv_out channels "
        "2 * H_k * D_k + H_v * D_v with H_v * D_v from recurrent_state.");
  }
  const int key_dim = (conv_dim - value_dim) / 2;
  if (key_dim % head_k_dim != 0) {
    return absl::UnavailableError(
        "gated_delta_update block mode expects the q / k width "
        "(conv_out channels - H_v * D_v) / 2 to be a multiple of D_k.");
  }
  const int num_k_heads = key_dim / head_k_dim;
  if (num_k_heads <= 0 || num_v_heads % num_k_heads != 0) {
    return absl::UnavailableError(
        "gated_delta_update block mode requires H_v to be a positive multiple "
        "of H_k.");
  }
  attr->num_k_heads = num_k_heads;
  attr->num_v_heads = num_v_heads;
  attr->head_k_dim = head_k_dim;
  attr->head_v_dim = head_v_dim;
  return absl::OkStatus();
}

absl::Status CheckGatedDeltaNetAttributes(const GatedDeltaNetAttributes& attr) {
  if (attr.num_k_heads <= 0 || attr.num_v_heads < attr.num_k_heads ||
      attr.num_v_heads % attr.num_k_heads != 0) {
    return absl::UnavailableError(
        "gated_delta_update block mode requires num_v_heads to be a positive "
        "multiple of num_k_heads.");
  }
  if (attr.head_k_dim != 128) {
    return absl::UnavailableError(
        "gated_delta_update block mode requires head_k_dim = 128.");
  }
  if (attr.head_v_dim < 16 || attr.head_v_dim > 128 ||
      attr.head_v_dim % 16 != 0) {
    return absl::UnavailableError(
        "gated_delta_update block mode requires head_v_dim to be a multiple of "
        "16 up to 128.");
  }
  if (attr.output_gate_type != "silu" && attr.output_gate_type != "swish" &&
      attr.output_gate_type != "sigmoid") {
    return absl::UnavailableError(
        "gated_delta_update block mode only supports output_gate_type 'silu' "
        "or 'sigmoid'.");
  }
  return absl::OkStatus();
}

absl::Status GatedDeltaUpdateOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (tflite_node->inputs->size != 6 && tflite_node->inputs->size != 7) {
    return absl::UnavailableError("gated_delta_update expects 6 or 7 inputs.");
  }
  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError("gated_delta_update expects 2 outputs.");
  }

  GatedDeltaUpdateAttributes parsed_attr =
      ParseGatedDeltaUpdateAttributes(*tflite_node, block_form_);
  if (parsed_attr.block.has_value()) {
    GatedDeltaNetAttributes& attr = *parsed_attr.block;
    for (int i = 0; i < tflite_node->inputs->size; ++i) {
      ABSL_RETURN_IF_ERROR(
          PreCheckRuntimeOrConstantInput(context, tflite_node, i));
    }
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const TfLiteTensor* conv_out = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &conv_out));
    const TfLiteTensor* zba = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &zba));
    const TfLiteTensor* a_log = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 2, &a_log));
    const TfLiteTensor* dt_bias = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 3, &dt_bias));
    const TfLiteTensor* norm_weight = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 4, &norm_weight));
    const TfLiteTensor* state = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 5, &state));
    if (!conv_out || !conv_out->dims || !zba || !zba->dims || !a_log ||
        !a_log->dims || !dt_bias || !dt_bias->dims || !norm_weight ||
        !norm_weight->dims || !state || !state->dims) {
      return absl::UnavailableError(
          "gated_delta_update block mode requires static input shapes.");
    }
    ABSL_RETURN_IF_ERROR(
        SetGatedDeltaNetDims(*conv_out->dims, *state->dims, &attr));
    ABSL_RETURN_IF_ERROR(CheckGatedDeltaNetAttributes(attr));

    const int key_dim = attr.num_k_heads * attr.head_k_dim;
    const int value_dim = attr.num_v_heads * attr.head_v_dim;
    auto num_elements = [](const TfLiteTensor* t) {
      int n = 1;
      for (int i = 0; i < t->dims->size; ++i) n *= t->dims->data[i];
      return n;
    };
    const TfLiteIntArray* conv_dims = conv_out->dims;
    const TfLiteIntArray* zba_dims = zba->dims;
    if (conv_dims->size != 3 || conv_dims->data[0] != 1 ||
        conv_dims->data[2] != 2 * key_dim + value_dim || zba_dims->size != 3 ||
        zba_dims->data[0] != 1 || zba_dims->data[1] != conv_dims->data[1] ||
        zba_dims->data[2] != value_dim + 2 * attr.num_v_heads) {
      return absl::UnavailableError(
          "gated_delta_update block mode expects conv_out [1, T, 2 * key_dim + "
          "value_dim] and zba [1, T, value_dim + 2 * H_v].");
    }
    if (num_elements(a_log) != attr.num_v_heads ||
        num_elements(dt_bias) != attr.num_v_heads ||
        num_elements(norm_weight) != attr.head_v_dim) {
      return absl::UnavailableError(
          "gated_delta_update block mode expects a_log [H_v], dt_bias [H_v], "
          "and norm_weight [D_v].");
    }
    if (state->type != kTfLiteFloat32 ||
        num_elements(state) != value_dim * attr.head_k_dim) {
      return absl::UnavailableError(
          "gated_delta_update block mode expects a float32 recurrent state "
          "[1, H_v, D_k, D_v].");
    }
    if (tflite_node->inputs->size == 7) {
      const TfLiteTensor* valid_mask = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetInputTensor(context, tflite_node, 6, &valid_mask));
      if (!valid_mask || !valid_mask->dims ||
          num_elements(valid_mask) != conv_dims->data[1]) {
        return absl::UnavailableError(
            "gated_delta_update block mode expects one valid_mask element per "
            "token.");
      }
    }
    return absl::OkStatus();
  }

  for (int i = 0; i < 6; ++i) {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, i));
  }
  if (tflite_node->inputs->size == 7) {
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 6));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  const TfLiteTensor* q = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &q));
  const TfLiteTensor* v = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 2, &v));
  if (q && q->dims && q->dims->size >= 4 && v && v->dims &&
      v->dims->size >= 4) {
    int H_k = q->dims->data[1];
    int H = v->dims->data[1];
    if (H_k <= 0 || H < H_k || (H % H_k != 0)) {
      return absl::InvalidArgumentError(
          "gated_delta_update requires H_v to be a positive multiple of H_k.");
    }
    int D_k = q->dims->data[q->dims->size - 1];
    int D_v = v->dims->data[v->dims->size - 1];
    bool d_k_valid =
        (D_k >= 16) && (D_k % 4 == 0) && (((D_k / 4) & ((D_k / 4) - 1)) == 0);
    bool d_v_valid =
        (D_v >= 16) && (D_v % 4 == 0) && (((D_v / 4) & ((D_v / 4) - 1)) == 0);
    if (!d_k_valid || !d_v_valid) {
      return absl::InvalidArgumentError(
          "gated_delta_update requires D_k and D_v to be powers of 2 (at "
          "least 16) and multiples of 4.");
    }
    if (tflite_node->inputs->size == 7) {
      const TfLiteTensor* valid_len = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetInputTensor(context, tflite_node, 6, &valid_len));
      if (valid_len && valid_len->dims) {
        int B = v->dims->data[0];
        int num_elems = 1;
        for (int i = 0; i < valid_len->dims->size; ++i) {
          num_elems *= valid_len->dims->data[i];
        }
        if (valid_len->dims->size == 0 || valid_len->dims->data[0] != B ||
            num_elems != B) {
          return absl::InvalidArgumentError(
              "gated_delta_update requires valid_len to have B elements with "
              "batch dimension B.");
        }
      }
    }
  }

  return absl::OkStatus();
}

void GatedDeltaUpdateOperationParser::Parse(const TfLiteNode* tflite_node,
                                            const TfLiteRegistration*,
                                            ::ml_drift::GraphFloat32* graph,
                                            ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kGatedDeltaUpdateType;
  GatedDeltaUpdateAttributes attr =
      ParseGatedDeltaUpdateAttributes(*tflite_node, block_form_);
  if (attr.block.has_value()) {
    // IsSupported has validated the shapes.
    const TfLiteTensor* conv_out = reader->GetInputTensor(0);
    const TfLiteTensor* state = reader->GetInputTensor(5);
    if (conv_out != nullptr && conv_out->dims != nullptr && state != nullptr &&
        state->dims != nullptr) {
      SetGatedDeltaNetDims(*conv_out->dims, *state->dims, &*attr.block)
          .IgnoreError();
    }
  }
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    // Skip input 5 (recurrent_state_in); we read and update output 1
    // (recurrent_state_out) in-place.
    if (i == 5) continue;
    if (attr.block.has_value() ? !reader->IsConstantTensor(i)
                               : reader->CanReadValue(i)) {
      reader->AddInput(node, i);
    } else {
      const ::ml_drift::Value* input = reader->AddConstInput(i, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    }
  }
  reader->AddOutputs(node);
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
