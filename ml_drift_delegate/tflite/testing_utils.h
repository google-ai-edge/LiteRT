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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_TESTING_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_TESTING_UTILS_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "tflite/interpreter.h"

namespace litert::ml_drift {

absl::Status FillTfLiteTensors(tflite::Interpreter* interpreter);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_TESTING_UTILS_H_
