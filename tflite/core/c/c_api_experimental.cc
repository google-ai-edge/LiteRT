/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/core/c/c_api_experimental.h"

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_internal.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/c/c_api.h"
#include "tflite/core/interpreter.h"
#include "tflite/profiling/telemetry/profiler.h"
#include "tflite/signature_runner.h"

extern "C" {

TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter) {
  return interpreter->impl->ResetVariableTensors();
}

int32_t TfLiteInterpreterGetVariableTensorCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int32_t>(interpreter->impl->variables().size());
}

TfLiteTensor* TfLiteInterpreterGetVariableTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
  return interpreter->impl->tensor(interpreter->impl->variables()[input_index]);
}

void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version) {
  options->mutable_op_resolver.AddBuiltin(
      static_cast<tflite::BuiltinOperator>(op), registration, min_version,
      max_version);
}

TfLiteInterpreter* TfLiteInterpreterCreateWithSelectedOps(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
  tflite::MutableOpResolver resolver;
  return tflite::internal::InterpreterCreateWithOpResolver(
      model, optional_options, &resolver);
}

void TfLiteInterpreterOptionsAddCustomOp(TfLiteInterpreterOptions* options,
                                         const char* name,
                                         const TfLiteRegistration* registration,
                                         int32_t min_version,
                                         int32_t max_version) {
  options->mutable_op_resolver.AddCustom(name, registration, min_version,
                                         max_version);
}

void TfLiteInterpreterOptionsSetOpResolverExternal(
    TfLiteInterpreterOptions* options,
    const TfLiteOperator* (*find_builtin_op)(void* user_data, int op,
                                             int version),
    const TfLiteOperator* (*find_custom_op)(void* user_data,
                                            const char* custom_op, int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_external = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op_external = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverExternalWithFallback(
    TfLiteInterpreterOptions* options,
    const TfLiteOperator* (*find_builtin_op_external)(void* user_data, int op,
                                                      int version),
    const TfLiteOperator* (*find_custom_op_external)(void* user_data,
                                                     const char* custom_op,
                                                     int version),
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data, const char* op,
                                                int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_external =
      find_builtin_op_external;
  options->op_resolver_callbacks.find_custom_op_external =
      find_custom_op_external;
  options->op_resolver_callbacks.find_builtin_op = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolver(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data, const char* op,
                                                int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverV1(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration_V1* (*find_builtin_op_v1)(void* user_data,
                                                       TfLiteBuiltinOperator op,
                                                       int version),
    const TfLiteRegistration_V1* (*find_custom_op_v1)(void* user_data,
                                                      const char* op,
                                                      int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_v1 = find_builtin_op_v1;
  options->op_resolver_callbacks.find_custom_op_v1 = find_custom_op_v1;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverV3(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration_V3* (*find_builtin_op_v3)(void* user_data,
                                                       TfLiteBuiltinOperator op,
                                                       int version),
    const TfLiteRegistration_V3* (*find_custom_op_v3)(void* user_data,
                                                      const char* op,
                                                      int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_v3 = find_builtin_op_v3;
  options->op_resolver_callbacks.find_custom_op_v3 = find_custom_op_v3;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverV2(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration_V2* (*find_builtin_op_v2)(void* user_data,
                                                       TfLiteBuiltinOperator op,
                                                       int version),
    const TfLiteRegistration_V2* (*find_custom_op_v2)(void* user_data,
                                                      const char* op,
                                                      int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_v2 = find_builtin_op_v2;
  options->op_resolver_callbacks.find_custom_op_v2 = find_custom_op_v2;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetUseNNAPI(TfLiteInterpreterOptions* options,
                                         bool enable) {
  options->use_nnapi = enable;
}

void TfLiteInterpreterOptionsSetEnableDelegateFallback(
    TfLiteInterpreterOptions* options, bool enable) {
  options->enable_delegate_fallback = enable;
}

TfLiteStatus TfLiteInterpreterModifyGraphWithDelegate(
    const TfLiteInterpreter* interpreter, TfLiteDelegate* delegate) {
  return interpreter->impl->ModifyGraphWithDelegate(delegate);
}

int32_t TfLiteInterpreterGetInputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
  return interpreter->impl->inputs()[input_index];
}

int32_t TfLiteInterpreterGetOutputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
  return interpreter->impl->outputs()[output_index];
}

int32_t TfLiteInterpreterGetSignatureCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int32_t>(interpreter->impl->signature_keys().size());
}

TfLiteStatus TfLiteInterpreterSetBufferHandle(TfLiteInterpreter* interpreter,
                                              TfLiteTensor* tensor,
                                              TfLiteBufferHandle buffer_handle,
                                              TfLiteOpaqueDelegate* delegate) {
  return interpreter->impl->SetBufferHandle(tensor, buffer_handle, delegate);
}

TfLiteStatus TfLiteInterpreterGetBufferHandle(TfLiteInterpreter* interpreter,
                                              int tensor_index,
                                              TfLiteBufferHandle* buffer_handle,
                                              TfLiteOpaqueDelegate** delegate) {
  return interpreter->impl->GetBufferHandle(tensor_index, buffer_handle,
                                            delegate);
}

void TfLiteSetAllowBufferHandleOutput(const TfLiteInterpreter* interpreter,
                                      bool allow_buffer_handle_output) {
  interpreter->impl->SetAllowBufferHandleOutput(allow_buffer_handle_output);
}

TfLiteStatus TfLiteInterpreterSetCustomAllocationForTensor(
    TfLiteInterpreter* interpreter, int tensor_index,
    const TfLiteCustomAllocation* allocation, int64_t flags) {
  if (allocation == nullptr) {
    return kTfLiteError;
  }
  return interpreter->impl->SetCustomAllocationForTensor(tensor_index,
                                                         *allocation, flags);
}

TfLiteStatus TfLiteInterpreterEnsureTensorDataIsReadable(
    TfLiteInterpreter* interpreter, int tensor_index) {
  return interpreter->impl->EnsureTensorDataIsReadable(tensor_index);
}

const char* TfLiteInterpreterGetSignatureKey(
    const TfLiteInterpreter* interpreter, int32_t signature_index) {
  int32_t signature_count = TfLiteInterpreterGetSignatureCount(interpreter);
  if (signature_index < 0 || signature_index >= signature_count) {
    return nullptr;
  }
  return interpreter->impl->signature_keys()[signature_index]->c_str();
}

TfLiteSignatureRunner* TfLiteInterpreterGetSignatureRunner(
    const TfLiteInterpreter* interpreter, const char* signature_key) {
  tflite::SignatureRunner* signature_runner =
      interpreter->impl->GetSignatureRunner(signature_key);
  if (!signature_runner) return nullptr;
  return new TfLiteSignatureRunner{signature_runner};
}

size_t TfLiteSignatureRunnerGetInputCount(
    const TfLiteSignatureRunner* signature_runner) {
  return signature_runner->impl->input_size();
}

const char* TfLiteSignatureRunnerGetInputName(
    const TfLiteSignatureRunner* signature_runner, const int32_t input_index) {
  int32_t input_count = TfLiteSignatureRunnerGetInputCount(signature_runner);
  if (input_index < 0 || input_index >= input_count) {
    return nullptr;
  }
  return signature_runner->impl->input_names()[input_index];
}

TfLiteStatus TfLiteSignatureRunnerResizeInputTensor(
    TfLiteSignatureRunner* signature_runner, const char* input_name,
    const int* input_dims, int32_t input_dims_size) {
  std::vector<int> dims{input_dims, input_dims + input_dims_size};
  return signature_runner->impl->ResizeInputTensorStrict(input_name, dims);
}

TfLiteStatus TfLiteSignatureRunnerAllocateTensors(
    TfLiteSignatureRunner* signature_runner) {
  return signature_runner->impl->AllocateTensors();
}

TfLiteTensor* TfLiteSignatureRunnerGetInputTensor(
    TfLiteSignatureRunner* signature_runner, const char* input_name) {
  return signature_runner->impl->input_tensor(input_name);
}

TfLiteStatus TfLiteSignatureRunnerInvoke(
    TfLiteSignatureRunner* signature_runner) {
  return signature_runner->impl->Invoke();
}

size_t TfLiteSignatureRunnerGetOutputCount(
    const TfLiteSignatureRunner* signature_runner) {
  return signature_runner->impl->output_size();
}

const char* TfLiteSignatureRunnerGetOutputName(
    const TfLiteSignatureRunner* signature_runner, int32_t output_index) {
  int32_t output_count = TfLiteSignatureRunnerGetOutputCount(signature_runner);
  if (output_index < 0 || output_index >= output_count) {
    return nullptr;
  }
  return signature_runner->impl->output_names()[output_index];
}

const TfLiteTensor* TfLiteSignatureRunnerGetOutputTensor(
    const TfLiteSignatureRunner* signature_runner, const char* output_name) {
  return signature_runner->impl->output_tensor(output_name);
}

TfLiteStatus TfLiteSignatureRunnerCancel(
    TfLiteSignatureRunner* signature_runner) {
  return signature_runner->impl->Cancel();
}

void TfLiteSignatureRunnerDelete(TfLiteSignatureRunner* signature_runner) {
  delete signature_runner;
}

void TfLiteInterpreterOptionsSetTelemetryProfiler(
    TfLiteInterpreterOptions* options,
    TfLiteTelemetryProfilerStruct* profiler) {
  options->telemetry_profiler = profiler;
}

}  // extern "C"
