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

#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_parser.h"

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qsa_parser_utils.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kQIdx = 0;
constexpr int kKIdxNew = 1;
constexpr int kRingK = kQsaIndexerRingKInput;
constexpr int kBlockKPool = kQsaIndexerBlockKPoolInput;
constexpr int kKNormWeight = 4;
constexpr int kBlockCos = 5;
constexpr int kBlockSin = 6;
constexpr int kStartPos = 7;
constexpr int kAttentionMask = 8;
constexpr int kNumInputs = 9;
constexpr int kNumOutputs = 3;

}  // namespace

absl::StatusOr<QsaIndexerUpdateAttributes> ParseQsaIndexerUpdateAttributes(
    const TfLiteNode& tflite_node) {
  QsaIndexerUpdateAttributes attr;
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
    if (!attr_map["rms_norm_eps"].IsNull()) {
      attr.rms_norm_eps = attr_map["rms_norm_eps"].AsFloat();
    }
    if (!attr_map["scale"].IsNull()) {
      attr.scale = attr_map["scale"].AsFloat();
    }
  }
  return attr;
}

absl::Status QsaIndexerUpdateOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (context == nullptr || tflite_node == nullptr ||
      tflite_node->inputs == nullptr || tflite_node->outputs == nullptr ||
      tflite_node->inputs->size != kNumInputs ||
      tflite_node->outputs->size != kNumOutputs) {
    return absl::UnavailableError(
        "odml.qsa_indexer_update expects 9 inputs and 3 outputs.");
  }
  for (int i : {kQIdx, kKIdxNew, kRingK, kBlockKPool}) {
    ABSL_RETURN_IF_ERROR(PreCheckQsaRuntimeOnlyInput(context, tflite_node, i));
  }
  for (int i :
       {kKNormWeight, kBlockCos, kBlockSin, kStartPos, kAttentionMask}) {
    ABSL_RETURN_IF_ERROR(
        PreCheckQsaRuntimeOrConstantInput(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckQsaOutputs(context, tflite_node));

  ABSL_ASSIGN_OR_RETURN(const QsaIndexerUpdateAttributes attr,
                        ParseQsaIndexerUpdateAttributes(*tflite_node));
  if (attr.compress_ratio != 4 || attr.block_topk <= 0 ||
      attr.block_topk > 2048) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update requires compress_ratio == 4 and "
        "block_topk in (0, 2048].");
  }

  auto input = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->inputs->data[i]];
  };
  auto output = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->outputs->data[i]];
  };

  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kQIdx), 4, kQsaIndexerUpdateCompositeName,
                         "q_idx"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kKIdxNew), 3, kQsaIndexerUpdateCompositeName,
                         "k_idx_new"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kRingK), 3, kQsaIndexerUpdateCompositeName,
                         "ring_k"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kBlockKPool), 3, kQsaIndexerUpdateCompositeName,
                         "block_k_pool"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kKNormWeight), 1, kQsaIndexerUpdateCompositeName,
                         "k_norm_weight"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kBlockCos), 3, kQsaIndexerUpdateCompositeName,
                         "block_cos"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kBlockSin), 3, kQsaIndexerUpdateCompositeName,
                         "block_sin"));
  ABSL_RETURN_IF_ERROR(CheckQsaScalarElementTensor(
      input(kStartPos), kQsaIndexerUpdateCompositeName, "start_pos"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(input(kAttentionMask), 4,
                         kQsaIndexerUpdateCompositeName, "attention_mask"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(output(0), 3, kQsaIndexerUpdateCompositeName,
                         "topk_block_indices"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(output(1), 3, kQsaIndexerUpdateCompositeName,
                         "new_ring_k"));
  ABSL_RETURN_IF_ERROR(
      CheckQsaTensorRank(output(2), 3, kQsaIndexerUpdateCompositeName,
                         "new_block_k_pool"));

  for (int i : {kQIdx, kKIdxNew, kRingK, kBlockKPool, kKNormWeight, kBlockCos,
                kBlockSin}) {
    if (!IsQsaFloatType(input(i).type)) {
      return absl::UnimplementedError(
          "odml.qsa_indexer_update expects float Q/K/norm/RoPE tensors.");
    }
  }
  if (!IsQsaFloatType(output(1).type) || !IsQsaFloatType(output(2).type)) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update expects float ring/pool outputs.");
  }
  if (input(kStartPos).type != kTfLiteInt32) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update expects a single-element int32 start_pos.");
  }
  if (input(kAttentionMask).type != kTfLiteBool &&
      !IsQsaFloatType(input(kAttentionMask).type)) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update expects bool or float attention_mask.");
  }
  if (output(0).type != kTfLiteInt32) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update expects int32 topk_block_indices output.");
  }

  const int batch_size = input(kQIdx).dims->data[0];
  const int seq_len = input(kQIdx).dims->data[1];
  const int idx_dim = input(kQIdx).dims->data[3];
  const int max_blocks = input(kBlockKPool).dims->data[1];
  const int rotary_dim = input(kBlockCos).dims->data[2];
  const int total_kv_len = input(kAttentionMask).dims->data[3];
  if (batch_size != 1) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update currently supports batch_size == 1.");
  }
  if (idx_dim <= 0 || idx_dim > 256 || idx_dim % 4 != 0) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update requires indexer_head_dim in (0, 256] and "
        "divisible by 4.");
  }
  if (rotary_dim <= 0 || rotary_dim > idx_dim || rotary_dim % 8 != 0) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update requires rotary_dim in (0, indexer_head_dim] "
        "and divisible by 8.");
  }
  if (max_blocks <= 0 || total_kv_len < max_blocks * attr.compress_ratio) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update requires max_blocks > 0 and attention_mask "
        "covering max_blocks * compress_ratio.");
  }
  if (input(kKIdxNew).dims->data[0] != 1 ||
      input(kKIdxNew).dims->data[1] != seq_len ||
      input(kKIdxNew).dims->data[2] != idx_dim ||
      input(kRingK).dims->data[0] != 1 ||
      input(kRingK).dims->data[1] != attr.compress_ratio ||
      input(kRingK).dims->data[2] != idx_dim ||
      input(kBlockKPool).dims->data[0] != 1 ||
      input(kBlockKPool).dims->data[2] != idx_dim ||
      input(kKNormWeight).dims->data[0] != idx_dim ||
      input(kBlockCos).dims->data[0] != 1 ||
      input(kBlockCos).dims->data[1] != max_blocks ||
      input(kBlockSin).dims->data[0] != 1 ||
      input(kBlockSin).dims->data[1] != max_blocks ||
      input(kBlockSin).dims->data[2] != rotary_dim ||
      input(kAttentionMask).dims->data[0] != 1 ||
      input(kAttentionMask).dims->data[1] != 1 ||
      input(kAttentionMask).dims->data[2] != seq_len) {
    return absl::InvalidArgumentError(
        "odml.qsa_indexer_update has inconsistent input dimensions.");
  }
  if (output(0).dims->data[0] != 1 || output(0).dims->data[1] != seq_len ||
      output(0).dims->data[2] != attr.block_topk ||
      output(1).dims->data[0] != 1 ||
      output(1).dims->data[1] != attr.compress_ratio ||
      output(1).dims->data[2] != idx_dim ||
      output(2).dims->data[0] != 1 || output(2).dims->data[1] != max_blocks ||
      output(2).dims->data[2] != idx_dim) {
    return absl::InvalidArgumentError(
        "odml.qsa_indexer_update has inconsistent output dimensions.");
  }
  return absl::OkStatus();
}

void QsaIndexerUpdateOperationParser::Parse(const TfLiteNode* tflite_node,
                                            const TfLiteRegistration*,
                                            ::ml_drift::GraphFloat32* graph,
                                            ObjectReader* reader) {
  ::ml_drift::Node* node = graph->NewNode();
  node->operation.type = kQsaIndexerUpdateType;
  // TFLite inputs 2 (`ring_k`) and 3 (`block_k_pool`) are omitted on GPU:
  // the kernel reads and updates outputs 1 (`new_ring_k`) and 2
  // (`new_block_k_pool`) in-place, matching `odml.short_conv_step` and
  // `odml.gated_delta_update` under `kGpuOptimizedInplace`.
  static constexpr int kSkippedStateInputs[] = {kRingK, kBlockKPool};
  AddQsaRuntimeOrConstInputs(*tflite_node, node, graph, reader,
                             kSkippedStateInputs);
  reader->AddOutputs(node);
  node->operation.attributes = ParseQsaIndexerUpdateAttributes(*tflite_node)
                                   .value_or(QsaIndexerUpdateAttributes{});
}

}  // namespace litert::ml_drift
