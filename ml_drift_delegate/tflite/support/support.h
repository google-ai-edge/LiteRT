// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

// Returns a list of supported node indices by ML Drift.
std::vector<int> GetSupportedNodes(
    TfLiteContext* absl_nonnull context,
    const IrModelBuilderOptions& options = IrModelBuilderOptions(),
    const CustomIrOpMap* custom_parsers = nullptr);

// Returns a list of supported node indices by ML Drift.
// Caller is responsible for TfLiteIntArrayFree the returned TfLiteIntArray.
TfLiteIntArray* GetOpsToReplace(TfLiteContext* absl_nonnull context,
                                const IrModelBuilderOptions& options,
                                const CustomIrOpMap* custom_parsers = nullptr);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_H_
