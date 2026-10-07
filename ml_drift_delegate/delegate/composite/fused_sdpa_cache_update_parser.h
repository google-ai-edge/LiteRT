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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSED_SDPA_CACHE_UPDATE_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSED_SDPA_CACHE_UPDATE_PARSER_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// Name of the StableHLO composite handled by this parser.
constexpr const char kFusedSdpaCacheUpdateCompositeName[] =
    "odml.fused_sdpa_cache_update";
// ML Drift node type of the parsed composite.
constexpr const char kFusedSdpaCacheUpdateType[] = "fused_sdpa_cache_update";

// `odml.fused_sdpa_cache_update` attends over a ring-buffer KV cache *before*
// this step's write plus the new tokens, and optionally writes the new tokens
// into the ring buffer. The node consumes 7 inputs:
//   0 query        [1, Hkv, G*T, D]  pre-scaled, query heads packed g-major.
//   1 key_cache    [1, Hkv, W, D]    packed kOSpatialIOGroupO4I4 GPU layout.
//   2 value_cache  [1, Hkv, D, W]    packed kOSpatialIOGroupI4O4 GPU layout.
//   3 key_new      [1, Hkv, T, D]
//   4 value_new    [1, Hkv, D, T]
//   5 mask         [1, 1, T, W + T]  bool (true = attend) or additive float.
//   6 param        int32, [0] = start position, [1] = end position.
// and produces 1 (attention) or 3 (attention, key_cache', value_cache')
// outputs. The updated caches are expected to be bound in place to the input
// cache buffers, so only the ring slots written by this step are stored.
struct FusedSdpaCacheUpdateAttributes {
  std::optional<float> softcap;
  // True when the node has 3 outputs and must write the ring buffer.
  bool update_cache = false;
};

class FusedSdpaCacheUpdateOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSED_SDPA_CACHE_UPDATE_PARSER_H_
