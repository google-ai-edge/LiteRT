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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_QSA_INDEXER_UPDATE_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_QSA_INDEXER_UPDATE_PARSER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kQsaIndexerUpdateCompositeName[] =
    "odml.qsa_indexer_update";
constexpr const char kQsaIndexerUpdateType[] = "qsa_indexer_update";

// TFLite input indices omitted on GPU because the kernel reads and updates
// outputs 1 (`new_ring_k`) and 2 (`new_block_k_pool`) in-place.
inline constexpr int kQsaIndexerRingKInput = 2;
inline constexpr int kQsaIndexerBlockKPoolInput = 3;
// GPU input index of `start_pos` after omitting `ring_k` and `block_k_pool`.
inline constexpr int kQsaIndexerStartPosGpuInput = 5;

// `odml.qsa_indexer_update` incrementally updates the QSA incomplete-block
// ring buffer (`ring_k`) and compressed block pool (`block_k_pool`), scores
// `q_idx` against complete blocks, and returns `topk_block_indices`.
//
// Inputs:
//   0 q_idx           [B, S, H_idx, D_idx]
//   1 k_idx_new       [B, S, D_idx]
//   2 ring_k          [B, R, D_idx]
//   3 block_k_pool    [B, max_blocks, D_idx]
//   4 k_norm_weight   [D_idx] (runtime or constant)
//   5 block_cos       [1, max_blocks, rotary_dim] (runtime or constant)
//   6 block_sin       [1, max_blocks, rotary_dim] (runtime or constant)
//   7 start_pos       [1] or scalar (int32, runtime or constant)
//   8 attention_mask  [B_m, 1, S, total_kv_len] (bool or float)
// Outputs:
//   0 topk_block_indices  [B, S, block_topk] (int32)
//   1 new_ring_k          [B, R, D_idx]
//   2 new_block_k_pool    [B, max_blocks, D_idx]
struct QsaIndexerUpdateAttributes {
  int compress_ratio = 4;
  int block_topk = 512;
  float rms_norm_eps = 1e-6f;
  float scale = 1.0f;
};

// Parses composite attributes, returning defaults when attributes are absent
// and `InvalidArgumentError` if the FlexBuffers buffer is malformed or does not
// have a map root.
absl::StatusOr<QsaIndexerUpdateAttributes> ParseQsaIndexerUpdateAttributes(
    const TfLiteNode& tflite_node);

class QsaIndexerUpdateOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_QSA_INDEXER_UPDATE_PARSER_H_
