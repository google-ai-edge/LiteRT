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

#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_parser.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/qsa_parser_utils.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kQuery = 0;
constexpr int kKeyCache = 1;
constexpr int kValueCache = 2;
constexpr int kTopkBlockIndices = 3;
constexpr int kAttentionMask = 4;
constexpr int kNumInputs = 5;
constexpr int kNumOutputs = 1;

}  // namespace

absl::StatusOr<SparseSdpaTransposedAttributes>
ParseSparseSdpaTransposedAttributes(const TfLiteNode& tflite_node) {
  SparseSdpaTransposedAttributes attr;
  ABSL_ASSIGN_OR_RETURN(const std::optional<flexbuffers::Map> fb_map,
                        GetQsaCompositeAttrMap(tflite_node));
  if (fb_map.has_value()) {
    const flexbuffers::Map& attr_map = *fb_map;
    if (!attr_map["compress_ratio"].IsNull()) {
      attr.compress_ratio = attr_map["compress_ratio"].AsInt32();
    }
    if (!attr_map["block_topk"].IsNull()) {
      attr.block_topk = attr_map["block_topk"].AsInt32();
    }
    if (!attr_map["scale"].IsNull()) {
      attr.scale = attr_map["scale"].AsFloat();
    }
    if (!attr_map["k_ts_idx"].IsNull()) {
      attr.k_ts_idx = attr_map["k_ts_idx"].AsInt32();
    }
    if (!attr_map["v_ts_idx"].IsNull()) {
      attr.v_ts_idx = attr_map["v_ts_idx"].AsInt32();
    }
    if (!attr_map["from_cache_update"].IsNull()) {
      attr.from_cache_update = attr_map["from_cache_update"].AsBool();
    }
  }
  return attr;
}

absl::Status SparseSdpaTransposedOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (context == nullptr || tflite_node == nullptr ||
      tflite_node->inputs == nullptr || tflite_node->outputs == nullptr ||
      tflite_node->inputs->size != kNumInputs ||
      tflite_node->outputs->size != kNumOutputs) {
    return absl::UnavailableError(
        "odml.sparse_sdpa_transposed expects 5 inputs and 1 output.");
  }
  for (int i : {kQuery, kKeyCache, kValueCache}) {
    ABSL_RETURN_IF_ERROR(PreCheckQsaRuntimeOnlyInput(context, tflite_node, i));
  }
  for (int i : {kTopkBlockIndices, kAttentionMask}) {
    ABSL_RETURN_IF_ERROR(
        PreCheckQsaRuntimeOrConstantInput(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckQsaOutputs(context, tflite_node));

  ABSL_ASSIGN_OR_RETURN(const SparseSdpaTransposedAttributes attr,
                        ParseSparseSdpaTransposedAttributes(*tflite_node));
  if (attr.compress_ratio != 4 || attr.block_topk <= 0 || attr.k_ts_idx != 2 ||
      attr.v_ts_idx != 3) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed requires compress_ratio == 4, "
        "block_topk > 0, k_ts_idx == 2, and v_ts_idx == 3.");
  }

  auto input = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->inputs->data[i]];
  };
  auto output = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->outputs->data[i]];
  };

  ABSL_RETURN_IF_ERROR(CheckQsaTensorRank(
      input(kQuery), 4, kSparseSdpaTransposedCompositeName, "query"));
  ABSL_RETURN_IF_ERROR(CheckQsaTensorRank(
      input(kKeyCache), 4, kSparseSdpaTransposedCompositeName, "key_cache"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kValueCache), 4,
                         kSparseSdpaTransposedCompositeName, "value_cache"));
  ABSL_RETURN_IF_ERROR(CheckQsaTensorRank(
      input(kTopkBlockIndices), 3, kSparseSdpaTransposedCompositeName,
      "topk_block_indices"));
  ABSL_RETURN_IF_ERROR(CheckQsaTensorRank(input(kAttentionMask), 4,
                                          kSparseSdpaTransposedCompositeName,
                                          "attention_mask"));
  ABSL_RETURN_IF_ERROR(CheckQsaTensorRank(
      output(0), 4, kSparseSdpaTransposedCompositeName, "attn_out"));

  for (int i : {kQuery, kKeyCache, kValueCache}) {
    if (!IsQsaFloatType(input(i).type)) {
      return absl::UnimplementedError(
          "odml.sparse_sdpa_transposed expects float Q/K/V tensors.");
    }
  }
  if (!IsQsaFloatType(output(0).type)) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed expects float attn_out tensor.");
  }
  if (input(kTopkBlockIndices).type != kTfLiteInt32) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed expects int32 topk_block_indices.");
  }
  if (input(kAttentionMask).type != kTfLiteBool &&
      !IsQsaFloatType(input(kAttentionMask).type)) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed expects bool or float attention_mask.");
  }

  const int32_t* q = input(kQuery).dims->data;
  const int32_t* kc = input(kKeyCache).dims->data;
  const int32_t* vc = input(kValueCache).dims->data;
  const int32_t* topk = input(kTopkBlockIndices).dims->data;
  const int32_t* mask = input(kAttentionMask).dims->data;
  const int32_t* out = output(0).dims->data;

  const int num_q_heads = q[1];
  const int seq_len = q[2];
  const int head_dim = q[3];
  const int num_kv_heads = kc[1];
  const int cache_len = kc[2];

  if (q[0] != 1 || kc[0] != 1 || vc[0] != 1 || topk[0] != 1 || mask[0] != 1 ||
      out[0] != 1) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed currently supports batch_size == 1.");
  }
  if (head_dim <= 0 || head_dim > 256 || head_dim % 4 != 0) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed requires head_dim in (0, 256] and "
        "divisible by 4.");
  }
  if (cache_len <= 0 || cache_len % 4 != 0) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed requires cache_len > 0 and divisible "
        "by 4.");
  }
  if (num_kv_heads <= 0 || num_q_heads < num_kv_heads ||
      num_q_heads % num_kv_heads != 0) {
    return absl::InvalidArgumentError(
        "odml.sparse_sdpa_transposed requires num_q_heads divisible by "
        "num_kv_heads.");
  }
  if (kc[3] != head_dim || vc[1] != num_kv_heads || vc[2] != head_dim ||
      vc[3] != cache_len) {
    return absl::InvalidArgumentError(
        "odml.sparse_sdpa_transposed has inconsistent K/V cache shapes.");
  }
  if (topk[1] != seq_len || topk[2] != attr.block_topk) {
    return absl::InvalidArgumentError(
        "odml.sparse_sdpa_transposed topk_block_indices shape must match "
        "[1, S, block_topk].");
  }
  if (mask[1] != 1 || mask[2] != seq_len || mask[3] != cache_len) {
    return absl::InvalidArgumentError(
        "odml.sparse_sdpa_transposed attention_mask must have shape "
        "[1, 1, S, cache_len].");
  }
  if (out[1] != seq_len || out[2] != num_q_heads || out[3] != head_dim) {
    return absl::InvalidArgumentError(
        "odml.sparse_sdpa_transposed output must have shape [1, S, H_q, D].");
  }
  return absl::OkStatus();
}

void SparseSdpaTransposedOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration*,
    ::ml_drift::GraphFloat32* graph, ObjectReader* reader) {
  ::ml_drift::Node* node = graph->NewNode();
  node->operation.type = kSparseSdpaTransposedType;
  AddQsaRuntimeOrConstInputs(*tflite_node, node, graph, reader);
  reader->AddOutputs(node);

  SparseSdpaTransposedAttributes attr =
      ParseSparseSdpaTransposedAttributes(*tflite_node)
          .value_or(SparseSdpaTransposedAttributes{});
  const auto inputs = graph->FindInputs(node->id);
  if (inputs.size() > kKeyCache) {
    ::ml_drift::Node* key_producer = graph->FindProducer(inputs[kKeyCache]->id);
    if (key_producer != nullptr &&
        key_producer->operation.type == kAddValuesToCacheType) {
      attr.from_cache_update = true;
    }
  }
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
