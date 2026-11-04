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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SHORT_CONV_STEP_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SHORT_CONV_STEP_PARSER_H_

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kShortConvStepType[] = "odml.short_conv_step";

// Attributes of `odml.short_conv_step`. Its conv_state is channels-last
// [1, L - 1, H] and its conv_weight channels-last [1, L, H], with
// L = `conv_L_cache` taps and H channels.
struct ShortConvStepAttributes {
  int32_t conv_L_cache = 3;
  bool is_gated = true;
  bool use_silu = false;
  // The optional `channels_last` attribute; only channels-last (true or
  // absent) is supported.
  bool channels_last = true;
};

ShortConvStepAttributes ParseShortConvStepAttributes(
    const TfLiteNode& tflite_node);

// Checks that `tflite_node` is a channels-last `odml.short_conv_step` the GPU
// kernel supports: conv_L_cache in [2, 4], `channels_last` not false,
// conv_state (input 1) [1, L - 1, H], conv_weight (input 2) [1, L, H] and
// output 0 with H channels. Channels-first exports ([1, H, L - 1] state) are
// rejected so they fall back to the CPU decomposition.
absl::Status CheckShortConvStepSupported(const TfLiteContext* context,
                                         const TfLiteNode* tflite_node);

class ShortConvStepOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SHORT_CONV_STEP_PARSER_H_
