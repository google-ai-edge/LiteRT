// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_RMS_NORM_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_RMS_NORM_H_

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

// Converts a TFLite RMS_NORM node to ML Drift IR.
void ConvertRmsNorm(const TfLiteContext& context, const TfLiteNode& node,
                    const TfLiteRegistration& registration,
                    ::ml_drift::ir::TensorMap& tensor_map,
                    ::ml_drift::ir::IrModel& ir_model);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_RMS_NORM_H_
