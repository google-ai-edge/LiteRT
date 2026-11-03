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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_PARSER_H_

#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"

namespace litert::ml_drift {

constexpr const char kGatedDeltaUpdateType[] = "gated_delta_update";

// Attributes of the fused GatedDeltaNet block mode of `gated_delta_update`
// (Qwen3.5 linear attention after the short conv). Inputs:
//   0: conv_out [1, T, 2 * H_k * D_k + H_v * D_v]
//   1: zba (fused [z | b | a] projection) [1, T, H_v * D_v + 2 * H_v]
//   2: a_log [H_v]
//   3: dt_bias [H_v]
//   4: norm_weight [D_v]
//   5: recurrent_state [1, H_v, D_k, D_v] (float32, updated in place)
//   6: optional valid_mask [1, T]
// Outputs:
//   0: gated RMSNorm output [1, T, H_v * D_v]
//   1: new recurrent state [1, H_v, D_k, D_v]
struct GatedDeltaNetAttributes {
  int num_k_heads = 0;
  int num_v_heads = 0;
  int head_k_dim = 0;
  int head_v_dim = 0;
  // Epsilon of the q / k L2 norms.
  float eps = 1e-6f;
  // Epsilon of the gated RMSNorm.
  float norm_eps = 1e-6f;
};

// TODO(kimishpatel): Get rid of the mode flag and make recurrent vs. chunked
// an internal implementation detail of the kernel rather than an attribute.
struct GatedDeltaUpdateAttributes {
  int mode = 0;  // 0: recurrent, 1: chunked
  std::string state_dtype = "float32";
  // Set when `gated_delta_update` wraps the full post-conv GatedDeltaNet block
  // (`num_k_heads > 0` in the custom op options).
  std::optional<GatedDeltaNetAttributes> block;
};

// Reads the attributes of a `gated_delta_update` node.
GatedDeltaUpdateAttributes ParseGatedDeltaUpdateAttributes(
    const TfLiteNode& tflite_node);

// Checks that the GPU kernel supports `attr`: D_k = 128, D_v a multiple of 16
// up to 128 (at most 8 subgroups per workgroup) and H_v a multiple of H_k.
absl::Status CheckGatedDeltaNetAttributes(const GatedDeltaNetAttributes& attr);

class GatedDeltaUpdateOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) override;

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) override;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_PARSER_H_
