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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_MOE_EXPERTS_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_MOE_EXPERTS_PARSER_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/object_reader.h"
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

constexpr const char kMoeExpertsType[] = "moe_experts";

using MoeScaleTensor =
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>;

struct MoeExpertsAttributes {
  enum class WeightType {
    kFp32,
    kInt8,
  };

  int num_experts = 0;
  int num_active_experts = 0;
  int model_dim = 0;
  int hidden_dim = 0;
  WeightType weight_type = WeightType::kFp32;
  std::optional<MoeScaleTensor> ff_gate_scale;
  std::optional<MoeScaleTensor> ff1_scale;
  std::optional<MoeScaleTensor> linear_scale;
};

class MoeExpertsOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final;

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_MOE_EXPERTS_PARSER_H_
