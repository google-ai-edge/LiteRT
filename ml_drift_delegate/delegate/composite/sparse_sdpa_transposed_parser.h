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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SPARSE_SDPA_TRANSPOSED_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SPARSE_SDPA_TRANSPOSED_PARSER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kSparseSdpaTransposedCompositeName[] =
    "odml.sparse_sdpa_transposed";
constexpr const char kSparseSdpaTransposedType[] = "sparse_sdpa_transposed";

// `odml.sparse_sdpa_transposed` computes GQA attention over only the KV
// micro-blocks selected by `topk_block_indices` plus trailing incomplete-block
// tokens.
//
// Inputs:
//   0 query               [B, H_q, S, D]
//   1 key_cache           [B, H_kv, cache_len, D] (or seq at k_ts_idx)
//   2 value_cache         [B, H_kv, D, cache_len] (or seq at v_ts_idx)
//   3 topk_block_indices  [B, S, block_topk] (int32)
//   4 attention_mask      [B_m, 1, S, cache_len] (bool or float)
// Outputs:
//   0 attn_out            [B, S, H_q, D]
struct SparseSdpaTransposedAttributes {
  int compress_ratio = 4;
  int block_topk = 512;
  float scale = 1.0f;
  int k_ts_idx = 2;
  int v_ts_idx = 3;
  bool from_cache_update = false;
};

// Parses composite attributes, returning defaults when attributes are absent
// and `InvalidArgumentError` if the FlexBuffers buffer is malformed or does not
// have a map root.
absl::StatusOr<SparseSdpaTransposedAttributes>
ParseSparseSdpaTransposedAttributes(const TfLiteNode& tflite_node);

class SparseSdpaTransposedOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SPARSE_SDPA_TRANSPOSED_PARSER_H_
