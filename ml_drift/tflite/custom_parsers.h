// Copyright 2024 The ML Drift Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CUSTOM_PARSERS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CUSTOM_PARSERS_H_

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"

namespace litert::ml_drift {

// Returns a parser for the provided custom op.
std::unique_ptr<TFLiteOperationParser> NewCustomOperationParser(
    absl::string_view op_name);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CUSTOM_PARSERS_H_
