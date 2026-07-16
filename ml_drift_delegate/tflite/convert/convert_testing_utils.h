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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_TESTING_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_TESTING_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {

class SingleOpInterpreterBuilder {
 public:
  explicit SingleOpInterpreterBuilder(int32_t builtin_code, int version = 1);
  ~SingleOpInterpreterBuilder();

  void AddInput(TfLiteType type, const std::vector<int>& shape);
  void AddInputWithId(int tensor_id);
  void AddConstInput(TfLiteType type, const std::vector<int>& shape,
                     const std::vector<uint8_t>& data);
  void AddOutput(TfLiteType type, const std::vector<int>& shape);

  // Takes ownership of params, which must be allocated with malloc/calloc.
  void SetParameters(void* params);
  void SetCustomName(const char* name);
  // Takes ownership of data, which must be allocated with malloc/calloc.
  void SetCustomData(void* data, size_t size);

  // Default interpreter builder using the inputs from AddInput().
  std::unique_ptr<::tflite::Interpreter> Build();
  // Interpreter builder that allows the user to specify the tflite op inputs.
  // Useful if the user wants to have the same input tensor be consumed multiple
  // times by the op. Should be used RARELY.
  std::unique_ptr<::tflite::Interpreter> Build(const std::vector<int>& inputs);

 private:
  std::unique_ptr<::tflite::Interpreter> interpreter_;
  TfLiteRegistration registration_;
  void* params_ = nullptr;
  void* custom_data_ = nullptr;
  size_t custom_data_size_ = 0;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  int tensor_count_ = 0;
};

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_TESTING_UTILS_H_
