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

#include "litert/tools/culprit_finder/tflite_input_manager.h"

#include <cstring>
#include <utility>

#include "litert/c/internal/litert_logging.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/tools/utils.h"

namespace litert::tools {

TfLiteStatus TfliteInputManager::PrepareInputData() {
  if (interpreter_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Interpreter is null");
    return kTfLiteError;
  }
  inputs_data_.clear();
  for (int input_index : interpreter_->inputs()) {
    TfLiteTensor* input_tensor = interpreter_->tensor(input_index);
    if (input_tensor->type == kTfLiteString) {
      LITERT_LOG(LITERT_ERROR, "String input tensor is not supported");
      return kTfLiteError;
    }
    float low_range = 0;
    float high_range = 0;
    tflite::utils::GetDataRangesForType(input_tensor->type, &low_range,
                                        &high_range);

    tflite::utils::InputTensorData input_data =
        tflite::utils::CreateRandomTensorData(*input_tensor, low_range,
                                              high_range);
    inputs_data_.push_back(std::move(input_data));
  }
  return kTfLiteOk;
}

TfLiteStatus TfliteInputManager::SetInputTensors(
    tflite::Interpreter& interpreter) {
  for (int i = 0; i < interpreter.inputs().size(); ++i) {
    int input_index = interpreter.inputs()[i];
    TfLiteTensor* input_tensor = interpreter.tensor(input_index);
    if (input_tensor->type == kTfLiteString) {
      LITERT_LOG(LITERT_ERROR, "String input tensor is not supported");
      return kTfLiteError;
    }
    std::memcpy(input_tensor->data.raw, inputs_data_[i].data.get(),
                inputs_data_[i].bytes);
  }
  return kTfLiteOk;
}
}  // namespace litert::tools
