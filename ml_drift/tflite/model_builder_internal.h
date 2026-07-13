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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_INTERNAL_H_

#include <memory>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// Returns a new TFLiteOperationParser object which parses the TFLite operator
// in the given TfLiteRegistration object.
std::unique_ptr<TFLiteOperationParser> NewOperationParser(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    bool allow_quant_ops = false,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops = nullptr);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_INTERNAL_H_
