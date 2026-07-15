// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_PRELU_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_PRELU_H_

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertPrelu(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_PRELU_H_
