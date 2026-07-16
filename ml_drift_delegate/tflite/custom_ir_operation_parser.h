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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CUSTOM_IR_OPERATION_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CUSTOM_IR_OPERATION_PARSER_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

// A struct binding the two callbacks together for a single composite op.
struct CustomIrOpParser {
  // Callback to check if a specific TFLite node can be parsed into an IrOp.
  absl::AnyInvocable<absl::Status(const TfLiteContext*, const TfLiteNode*,
                                  const TfLiteRegistration*) const>
      is_supported;

  // Callback to convert the TFLite node into an IrOp.
  absl::AnyInvocable<void(
      const TfLiteContext&, const TfLiteNode&, const TfLiteRegistration&,
      absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>&,
      const IrModelBuilderOptions&, ::ml_drift::ir::IrModel&) const>
      convert;
};

// A registry mapping composite names (e.g., "odml.cache_update") to their
// parsers.
using CustomIrOpMap = absl::flat_hash_map<std::string, CustomIrOpParser>;

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CUSTOM_IR_OPERATION_PARSER_H_
