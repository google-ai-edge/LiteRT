/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tflite/core/interpreter.h"
#include "tflite/kernels/test_util.h"

namespace tflite {
namespace testing {
namespace {

// Reads as much from |data| as possible into |tensor->bytes|, padding the
// tensor buffer with 0 when the source data is exhausted.
void FillTensor(const uint8_t** data, size_t* bytes_remaining,
                TfLiteTensor* tensor) {
  memset(tensor->data.uint8, 0, tensor->bytes);
  const size_t bytes_to_copy = std::min(tensor->bytes, *bytes_remaining);
  if (bytes_to_copy) {
    memcpy(tensor->data.uint8, *data, bytes_to_copy);
    *data += bytes_to_copy;
    *bytes_remaining -= bytes_to_copy;
  }
}

// Utility class for creating an interpreter and model for use with fuzzing.
class TfLiteFuzzerContext : public SingleOpModel {
 public:
  TfLiteFuzzerContext() {
    // Create a simple model with an add operator.
    auto input1 = AddInput({TensorType_FLOAT32, {2, 2}});
    auto input2 = AddInput({TensorType_FLOAT32, {2, 2}});
    AddOutput({TensorType_FLOAT32, {}});
    SetBuiltinOp(
        BuiltinOperator_ADD, BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union());
    BuildInterpreter({GetShape(input1), GetShape(input2)});
  }

  Interpreter* interpreter() const { return interpreter_.get(); }
};

}  // namespace

// Simple test which fuzzes input tensor data before executing a simple model.
void SimpleFuzzerTest(const std::string& data) {
  // Loading the model and allocating tensors is relatively expensive, so create
  // a single interpreter and use it for multiple iterations.
  static TfLiteFuzzerContext* fuzzer_context = new TfLiteFuzzerContext{};

  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.data());
  size_t size = data.size();

  // Populate inputs with fuzzed data.
  Interpreter* interpreter = fuzzer_context->interpreter();
  for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
    auto input_tensor = interpreter->tensor(interpreter->inputs()[i]);
    FillTensor(&ptr, &size, input_tensor);
  }

  // Execute inference with the fuzzed inputs.
  fuzzer_context->interpreter()->Invoke();
}
FUZZ_TEST(SimpleFuzzer, SimpleFuzzerTest);

}  // namespace testing
}  // namespace tflite
