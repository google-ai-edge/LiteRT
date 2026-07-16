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

#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {

namespace {
TfLiteQuantization GetQuantization(TfLiteType type) {
  if (type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
      type == kTfLiteBFloat16) {
    return {kTfLiteNoQuantization, nullptr};
  }
  // For quantized types, we need kTfLiteAffineQuantization.
  auto* quant_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = TfLiteFloatArrayCreate(1);
  quant_params->scale->data[0] = 1.0f;
  quant_params->zero_point = TfLiteIntArrayCreate(1);
  quant_params->zero_point->data[0] = 0;
  quant_params->quantized_dimension = 0;
  return {kTfLiteAffineQuantization, quant_params};
}
}  // namespace

SingleOpInterpreterBuilder::SingleOpInterpreterBuilder(int builtin_code,
                                                       int version)
    : interpreter_(std::make_unique<::tflite::Interpreter>()) {
  registration_ = {nullptr, nullptr, nullptr, nullptr};
  registration_.builtin_code = builtin_code;
  registration_.version = version;
}

SingleOpInterpreterBuilder::~SingleOpInterpreterBuilder() {
  free(params_);
  free(custom_data_);
}

void SingleOpInterpreterBuilder::AddInput(TfLiteType type,
                                          const std::vector<int>& shape) {
  interpreter_->AddTensors(1);
  interpreter_->SetTensorParametersReadWrite(tensor_count_, type, "", shape,
                                             GetQuantization(type));
  inputs_.push_back(tensor_count_);
  tensor_count_++;
}

void SingleOpInterpreterBuilder::AddInputWithId(int tensor_id) {
  inputs_.push_back(tensor_id);
}

void SingleOpInterpreterBuilder::AddConstInput(
    TfLiteType type, const std::vector<int>& shape,
    const std::vector<uint8_t>& data) {
  interpreter_->AddTensors(1);
  interpreter_->SetTensorParametersReadOnly(
      tensor_count_, type, "", shape, GetQuantization(type),
      reinterpret_cast<const char*>(data.data()), data.size());
  inputs_.push_back(tensor_count_);
  tensor_count_++;
}

void SingleOpInterpreterBuilder::AddOutput(TfLiteType type,
                                           const std::vector<int>& shape) {
  interpreter_->AddTensors(1);
  interpreter_->SetTensorParametersReadWrite(tensor_count_, type, "", shape,
                                             GetQuantization(type));
  outputs_.push_back(tensor_count_);
  tensor_count_++;
}

void SingleOpInterpreterBuilder::SetParameters(void* params) {
  if (params_ != params) {
    free(params_);
    params_ = params;
  }
}
void SingleOpInterpreterBuilder::SetCustomName(const char* name) {
  registration_.custom_name = name;
}

void SingleOpInterpreterBuilder::SetCustomData(void* data, size_t size) {
  if (custom_data_ != data) {
    free(custom_data_);
    custom_data_ = data;
    custom_data_size_ = size;
  }
}

std::unique_ptr<::tflite::Interpreter> SingleOpInterpreterBuilder::Build() {
  return Build(inputs_);
}

std::unique_ptr<::tflite::Interpreter> SingleOpInterpreterBuilder::Build(
    const std::vector<int>& inputs) {
  interpreter_->SetInputs(inputs);
  interpreter_->SetOutputs(outputs_);
  interpreter_->AddNodeWithParameters(
      inputs, outputs_, reinterpret_cast<const char*>(custom_data_),
      custom_data_size_, params_, &registration_);
  params_ = nullptr;
  return std::move(interpreter_);
}

}  // namespace litert::ml_drift::ir
