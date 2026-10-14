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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_ADD_VALUES_TO_CACHE_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_ADD_VALUES_TO_CACHE_PARSER_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"

namespace litert::ml_drift {

constexpr const char kAddValuesToCacheType[] = "add_values_to_cache";

struct AddValuesToCacheAttributes {
  int kv_cache_batch_size;
  int cache_size;
  int head_size;
  // quantized kv cache case
  std::optional<float> scale_k;
  std::optional<float> scale_v;
  // Local Attention Ring buffer case
  std::optional<bool> is_ring_buffer;
  // Index of the time axis of each 4D tensor ([B, H, *, *], so 2 or 3). The
  // defaults are the layouts the op had before these attributes existed:
  // K cache [B, H, S, D], V cache [B, H, D, S], and both updates [B, H, T, D].
  int k_cache_ts_idx = 2;
  int v_cache_ts_idx = 3;
  int k_update_ts_idx = 2;
  int v_update_ts_idx = 2;
};

// Reads the optional `*_ts_idx` layout attributes of `odml.cache_update` into
// `attr` (absent attributes keep their defaults), and returns an Unavailable
// error for layouts the kernel does not implement so that the node is left to
// the CPU decomposition instead of being computed with the wrong layout.
absl::Status ReadAddValuesToCacheLayout(const flexbuffers::Map& attributes,
                                        AddValuesToCacheAttributes& attr);

class AddValuesToCacheOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_ADD_VALUES_TO_CACHE_PARSER_H_
