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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GLU_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GLU_PARSER_H_

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// Gated linear unit (GLU) composite: act(gate) * up. The composite keeps its
// original `odml.swiglu` name for compatibility with exported models, but the
// activation is selected by the `activation` attribute (SwiGLU or GeGLU).
constexpr const char kSwigluType[] = "odml.swiglu";

// Activation applied to the gate half before multiplying with the up half.
enum class GluActivation {
  kSilu = 0,      // SwiGLU: silu(gate) * up.
  kGeluTanh = 1,  // GeGLU: gelu_tanh(gate) * up (e.g. Gemma).
};

struct GluAttributes {
  int32_t gate_size = 0;
  GluActivation activation = GluActivation::kSilu;
};

// Parses the optional "activation" composite attribute. Unknown or missing
// values map to kSilu for backward compatibility.
GluActivation ParseGluActivation(absl::string_view activation);

class GluOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GLU_PARSER_H_
