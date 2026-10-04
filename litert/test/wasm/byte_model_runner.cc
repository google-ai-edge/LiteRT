// Copyright 2026 Google LLC.
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

// A comparison harness for byte-input classifiers. Both builds expose the
// same small C ABI, retain the model and buffers, and return every float
// output. It deliberately contains no tokenizer, fetch, UI, or serving policy.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef LITERT_WASM_TFLITE_BASELINE
#include <memory>

#include "litert/runtime/op_resolver.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/model_builder.h"
#else
#include <optional>
#include <utility>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"
#endif

namespace {

// Retained for the lifetime of the module and intentionally never destroyed:
// globals with static storage duration must be trivially destructible.
std::vector<uint8_t>& model_bytes = *new std::vector<uint8_t>;
std::vector<std::vector<float>>& output_values =
    *new std::vector<std::vector<float>>;
size_t input_length = 0;
bool ready = false;

#ifdef LITERT_WASM_TFLITE_BASELINE
std::unique_ptr<tflite::FlatBufferModel>& model =
    *new std::unique_ptr<tflite::FlatBufferModel>;
std::unique_ptr<tflite::Interpreter>& interpreter =
    *new std::unique_ptr<tflite::Interpreter>;
#else
std::optional<litert::Environment>& environment =
    *new std::optional<litert::Environment>;
std::optional<litert::CompiledModel>& compiled_model =
    *new std::optional<litert::CompiledModel>;
std::vector<litert::TensorBuffer>& inputs =
    *new std::vector<litert::TensorBuffer>;
std::vector<litert::TensorBuffer>& outputs =
    *new std::vector<litert::TensorBuffer>;
#endif

}  // namespace

extern "C" {

int load_model(const uint8_t* data, size_t size) {
  ready = false;
  input_length = 0;
  output_values.clear();
#ifdef LITERT_WASM_TFLITE_BASELINE
  interpreter.reset();
  model.reset();
#else
  inputs.clear();
  outputs.clear();
  compiled_model.reset();
  environment.reset();
#endif
  model_bytes.clear();
  if (!data || !size) return 1;
  model_bytes.assign(data, data + size);
#ifdef LITERT_WASM_TFLITE_BASELINE
  model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());
  if (!model) return 2;
  auto resolver = litert::internal::CreateOpResolver(false);
  if (!resolver) return 3;
  tflite::InterpreterBuilder builder(*model, **resolver);
  builder.SetNumThreads(1);
  if (builder(&interpreter) != kTfLiteOk ||
      interpreter->AllocateTensors() != kTfLiteOk) {
    return 4;
  }
  if (interpreter->inputs().size() != 1) return 5;
  auto* input = interpreter->input_tensor(0);
  if (input->type != kTfLiteInt32) return 6;
  input_length = input->bytes / sizeof(int32_t);
  for (int index : interpreter->outputs()) {
    const auto* output = interpreter->tensor(index);
    if (output->type != kTfLiteFloat32) return 7;
    output_values.emplace_back(output->bytes / sizeof(float));
  }
#else
  auto env = litert::Environment::Create({});
  if (!env) return 2;
  environment.emplace(std::move(*env));
  auto compiled = litert::CompiledModel::Create(
      *environment,
      litert::BufferRef<uint8_t>(model_bytes.data(), model_bytes.size()),
      litert::HwAccelerators::kCpu);
  if (!compiled) return 3;
  compiled_model.emplace(std::move(*compiled));
  auto in = compiled_model->CreateInputBuffers();
  auto out = compiled_model->CreateOutputBuffers();
  if (!in || !out) return 4;
  inputs = std::move(*in);
  outputs = std::move(*out);
  if (inputs.size() != 1) return 5;
  auto type = inputs[0].TensorType();
  auto bytes = inputs[0].PackedSize();
  if (!type || type->ElementType() != litert::ElementType::Int32 || !bytes) {
    return 6;
  }
  input_length = *bytes / sizeof(int32_t);
  for (auto& output : outputs) {
    auto type = output.TensorType();
    auto bytes = output.PackedSize();
    if (!type || type->ElementType() != litert::ElementType::Float32 ||
        !bytes) {
      return 7;
    }
    output_values.emplace_back(*bytes / sizeof(float));
  }
#endif
  ready = true;
  return 0;
}

size_t get_input_length() { return input_length; }

int predict(const int32_t* data, size_t length) {
  if (!ready || !data || length != input_length) return 1;
#ifdef LITERT_WASM_TFLITE_BASELINE
  std::memcpy(interpreter->typed_input_tensor<int32_t>(0), data,
              length * sizeof(int32_t));
  if (interpreter->Invoke() != kTfLiteOk) return 2;
  for (size_t i = 0; i < output_values.size(); ++i) {
    std::memcpy(output_values[i].data(),
                interpreter->typed_output_tensor<float>(i),
                output_values[i].size() * sizeof(float));
  }
#else
  if (!inputs[0].Write<int32_t>({data, length})) return 2;
  if (!compiled_model->Run(inputs, outputs)) return 3;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].Read<float>(
            {output_values[i].data(), output_values[i].size()}))
      return 4;
  }
#endif
  return 0;
}

size_t get_output_count() { return output_values.size(); }

size_t get_output_length(size_t index) {
  return index < output_values.size() ? output_values[index].size() : 0;
}

const float* get_output(size_t index) {
  return index < output_values.size() ? output_values[index].data() : nullptr;
}

}  // extern "C"
