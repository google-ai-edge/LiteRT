/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"

#include <cstring>
#include <memory>
#include <vector>

#include "tflite/core/interpreter.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/kernels/fuzzing/fuzzer_quota_allocator.h"
#include "tflite/model_builder.h"
#include "tflite/mutable_op_resolver.h"
#include "tflite/version.h"

namespace tflite {
namespace fuzzing {
namespace {

flatbuffers::Offset<OperatorCode> MakeOperatorCode(
    flatbuffers::FlatBufferBuilder* builder, BuiltinOperator op, int version) {
  if (static_cast<int>(op) >
      static_cast<int>(BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES)) {
    return CreateOperatorCode(
        *builder, BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES, 0, version,
        op);
  }
  return CreateOperatorCode(*builder, op, 0, version);
}

}  // namespace

RunResult BuildAndRunOneOpModel(flatbuffers::FlatBufferBuilder* builder,
                                const OneOpModelSpec& model_spec,
                                const OneOpRunSpec& run_spec) {
  if (builder == nullptr || run_spec.registration == nullptr ||
      run_spec.max_live_allocation_bytes == 0) {
    return RunResult::kHarnessFailure;
  }

  const auto opcode = MakeOperatorCode(
      builder, model_spec.builtin_operator, model_spec.version);
  const auto op = CreateOperator(
      *builder, /*opcode_index=*/0, builder->CreateVector(model_spec.op_inputs),
      builder->CreateVector(model_spec.op_outputs),
      model_spec.builtin_options_type, model_spec.builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS,
      /*mutating_variable_inputs=*/0, /*intermediates=*/0,
      /*large_custom_options_offset=*/0, /*large_custom_options_size=*/0,
      model_spec.builtin_options_2_type, model_spec.builtin_options_2);
  const auto subgraph = CreateSubGraph(
      *builder, builder->CreateVector(model_spec.tensors),
      builder->CreateVector(model_spec.model_inputs),
      builder->CreateVector(model_spec.model_outputs),
      builder->CreateVector(&op, 1));

  std::vector<flatbuffers::Offset<Buffer>> buffers = model_spec.buffers;
  if (buffers.empty()) {
    buffers.push_back(CreateAlignedBuffer(builder, std::vector<uint8_t>{}));
  }
  const auto model = CreateModel(
      *builder, TFLITE_SCHEMA_VERSION, builder->CreateVector(&opcode, 1),
      builder->CreateVector(&subgraph, 1),
      builder->CreateString(model_spec.description),
      builder->CreateVector(buffers));
  builder->Finish(model);
  if (!IsAligned(builder->GetBufferPointer(),
                 builder->GetBufferMinAlignment())) {
    return RunResult::kHarnessFailure;
  }

  const Model* model_view = GetModel(builder->GetBufferPointer());
  if (!ConstantTensorBuffersAreAligned(model_view)) {
    return RunResult::kHarnessFailure;
  }

  MutableOpResolver resolver;
  resolver.AddBuiltin(model_spec.builtin_operator, run_spec.registration,
                      run_spec.min_version, run_spec.max_version);
  SilentErrorReporter error_reporter;
  FuzzerQuotaAllocator quota_allocator(run_spec.max_live_allocation_bytes);
  std::unique_ptr<Interpreter> interpreter;
  if (InterpreterBuilder(model_view, resolver, &error_reporter)(&interpreter) !=
          kTfLiteOk ||
      interpreter == nullptr) {
    return RunResult::kRejected;
  }
  if (interpreter->SetAllocator(quota_allocator.allocator()) != kTfLiteOk) {
    return RunResult::kHarnessFailure;
  }

  for (const RuntimeTensor& runtime_tensor : run_spec.runtime_tensors) {
    if (runtime_tensor.shape.has_value() &&
        interpreter->ResizeInputTensor(runtime_tensor.tensor_index,
                                       *runtime_tensor.shape) != kTfLiteOk) {
      return RunResult::kRejected;
    }
  }
  for (const int tensor_index : run_spec.persistent_ro_tensors) {
    TfLiteTensor* tensor = interpreter->tensor(tensor_index);
    if (tensor == nullptr) {
      return RunResult::kHarnessFailure;
    }
    tensor->allocation_type = kTfLitePersistentRo;
    if (TfLiteTensorResizeMaybeCopyWithAllocator(
            tensor->bytes, tensor, /*preserve_data=*/false,
            quota_allocator.allocator()) != kTfLiteOk) {
      return RunResult::kRejected;
    }
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return RunResult::kRejected;
  }
  if (run_spec.post_allocate) {
    const RunResult hook_result = run_spec.post_allocate(interpreter.get());
    if (hook_result != RunResult::kSuccess) {
      return hook_result;
    }
  }

  for (const RuntimeTensor& runtime_tensor : run_spec.runtime_tensors) {
    TfLiteTensor* tensor = interpreter->tensor(runtime_tensor.tensor_index);
    if (tensor == nullptr) {
      return RunResult::kHarnessFailure;
    }
    if (!TensorDataIsAligned(tensor)) {
      return RunResult::kHarnessFailure;
    }
    if (!runtime_tensor.data.has_value()) {
      continue;
    }
    if (tensor->bytes != runtime_tensor.data->size()) {
      return RunResult::kRejected;
    }
    if (!runtime_tensor.data->empty()) {
      if (tensor->data.raw == nullptr) {
        return RunResult::kRejected;
      }
      std::memcpy(tensor->data.raw, runtime_tensor.data->data(),
                  runtime_tensor.data->size());
    }
  }

  if (!run_spec.invoke) {
    return RunResult::kSuccess;
  }
  return interpreter->Invoke() == kTfLiteOk ? RunResult::kSuccess
                                            : RunResult::kRejected;
}

}  // namespace fuzzing
}  // namespace tflite
