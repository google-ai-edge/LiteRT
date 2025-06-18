// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_TOOLS_CULPRIT_FINDER_TFLITE_INPUT_MANAGER_H_
#define ODML_LITERT_TOOLS_CULPRIT_FINDER_TFLITE_INPUT_MANAGER_H_
#include <vector>

#include "tflite/interpreter.h"
#include "tflite/tools/utils.h"

namespace litert::tools {

// TfliteInputManager is a class that manages the input data for the Tflite
// model. It provides methods to prepare random input data and set the input
// tensors. The interpreter is not owned by the TfliteInputManager and must
// outlive the TfliteInputManager.
class TfliteInputManager {
 public:
  explicit TfliteInputManager(tflite::Interpreter* interpreter)
      : interpreter_(interpreter) {};

  // Prepares random input data for the Tflite model. Resets if the input data
  // is already prepared.
  TfLiteStatus PrepareInputData();

  // Sets the input tensors for the Passed interpreter. We allow passing in the
  // interpreter for delegated models.
  TfLiteStatus SetInputTensors(tflite::Interpreter& interpreter);

 private:
  tflite::Interpreter* interpreter_;
  std::vector<tflite::utils::InputTensorData> inputs_data_;
};
}  // namespace litert::tools

#endif  // ODML_LITERT_TOOLS_CULPRIT_FINDER_TFLITE_INPUT_MANAGER_H_
