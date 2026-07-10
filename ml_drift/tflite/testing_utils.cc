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

#include "third_party/odml/litert/ml_drift/tflite/testing_utils.h"

#include <cstdint>
#include <random>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {

absl::Status FillTfLiteTensors(tflite::Interpreter* interpreter) {
  for (int i = 0; i < interpreter->inputs().size(); ++i) {
    std::mt19937 gen(i);
    TfLiteTensor* tensor_ptr = interpreter->tensor(interpreter->inputs()[i]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    switch (tensor_ptr->type) {
      case kTfLiteFloat32:
        ::ml_drift::GenerateData(gen, interpreter->typed_input_tensor<float>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteInt32:
        ::ml_drift::GenerateData(gen,
                                 interpreter->typed_input_tensor<int32_t>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteInt16:
        ::ml_drift::GenerateData(gen,
                                 interpreter->typed_input_tensor<int16_t>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteInt8:
        ::ml_drift::GenerateData(gen,
                                 interpreter->typed_input_tensor<int8_t>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteUInt32:
        ::ml_drift::GenerateData(gen,
                                 interpreter->typed_input_tensor<uint32_t>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteUInt16:
        ::ml_drift::GenerateData(gen,
                                 interpreter->typed_input_tensor<uint16_t>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteUInt8:
        ::ml_drift::GenerateData(gen,
                                 interpreter->typed_input_tensor<uint8_t>(i),
                                 tensor_elements_count);
        break;
      case kTfLiteBool:
        ::ml_drift::GenerateData(gen, interpreter->typed_input_tensor<bool>(i),
                                 tensor_elements_count);
        break;
      default:
        return absl::InvalidArgumentError("Unsupported tensor type");
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::ml_drift
