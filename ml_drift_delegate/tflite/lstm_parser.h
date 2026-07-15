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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_LSTM_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_LSTM_PARSER_H_

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

void ParseLSTMAttributes(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    ::ml_drift::GraphFloat32* graph, ObjectReader* reader,
    const TfLiteLSTMParams* params,
    absl::flat_hash_map<int, ::ml_drift::ValueId>* new_variable_input_values);
}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_LSTM_PARSER_H_
