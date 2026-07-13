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

#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/util.h"

namespace litert::ml_drift::ir {
namespace {

// TODO: who/impjdi - This is not a safe solution; find a better way to stub
// TfLiteContext without this hack.
TfLiteIntArray* g_execution_plan;
TfLiteNode* g_node;
TfLiteRegistration* g_registration;
TfLiteDelegateParams* g_delegate_params;

TfLiteIntArray* CopySpanToTfLiteIntArray(absl::Span<const int> span) {
  TfLiteIntArray* array = TfLiteIntArrayCreate(span.size());
  std::copy(span.begin(), span.end(), array->data);
  return array;
}

TfLiteStatus GetExecutionPlan(TfLiteContext*, TfLiteIntArray** execution_plan) {
  *execution_plan = g_execution_plan;
  return kTfLiteOk;
}

TfLiteStatus GetNodeAndRegistration(TfLiteContext*, int, TfLiteNode** node,
                                    TfLiteRegistration** registration) {
  *node = g_node;
  *registration = g_registration;
  return kTfLiteOk;
}

TfLiteStatus PreviewDelegatePartitioning(
    TfLiteContext*, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegateParams** partition_params_array, int* num_partitions) {
  TfLiteIntArrayFree(g_delegate_params->nodes_to_replace);
  g_delegate_params->nodes_to_replace = TfLiteIntArrayCopy(nodes_to_replace);
  *partition_params_array = g_delegate_params;
  *num_partitions = 1;
  return kTfLiteOk;
}

}  // namespace

StubContextBuilder::StubContextBuilder() {
  g_execution_plan = TfLiteIntArrayCreate(1);
  g_execution_plan->data[0] = 0;
  g_node = new TfLiteNode;
  std::memset(g_node, 0, sizeof(TfLiteNode));
  g_registration = new TfLiteRegistration;
  std::memset(g_registration, 0, sizeof(TfLiteRegistration));
  g_delegate_params = new TfLiteDelegateParams;
  std::memset(g_delegate_params, 0, sizeof(TfLiteDelegateParams));
  std::memset(&context_, 0, sizeof(TfLiteContext));
  context_.GetExecutionPlan = GetExecutionPlan;
  context_.ReportError = NullReportError;  // LoggingReportError to see errors.
  context_.GetNodeAndRegistration = GetNodeAndRegistration;
  context_.PreviewDelegatePartitioning = PreviewDelegatePartitioning;
}

StubContextBuilder::~StubContextBuilder() {
  TfLiteIntArrayFree(g_delegate_params->nodes_to_replace);
  TfLiteIntArrayFree(g_delegate_params->output_tensors);
  TfLiteIntArrayFree(g_delegate_params->input_tensors);
  delete g_delegate_params;
  for (TfLiteTensor& tensor : tensors_) {
    TfLiteIntArrayFree(tensor.dims);
    if (tensor.quantization.params) {
      TfLiteAffineQuantization* affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(
              tensor.quantization.params);
      TfLiteFloatArrayFree(affine_quantization->scale);
      TfLiteIntArrayFree(affine_quantization->zero_point);
      delete affine_quantization;
      tensor.quantization.params = nullptr;
    }
    if (tensor.sparsity) {
      delete tensor.sparsity;
      tensor.sparsity = nullptr;
    }
  }
  delete g_registration;
  TfLiteIntArrayFree(g_node->outputs);
  TfLiteIntArrayFree(g_node->inputs);
  delete g_node;
  TfLiteIntArrayFree(g_execution_plan);
}

int StubContextBuilder::AddTensor(TfLiteType dtype,
                                  absl::Span<const int> dims) {
  TfLiteTensor& tensor = tensors_.emplace_back();
  std::memset(&tensor, 0, sizeof(TfLiteTensor));
  tensor.type = dtype;
  tensor.dims = CopySpanToTfLiteIntArray(dims);
  return tensors_.size() - 1;
}

int StubContextBuilder::AddQuantizedTensor(TfLiteType dtype,
                                           absl::Span<const int> dims,
                                           int scale_size) {
  TfLiteTensor& tensor = tensors_.emplace_back();
  std::memset(&tensor, 0, sizeof(TfLiteTensor));
  tensor.type = dtype;
  tensor.dims = CopySpanToTfLiteIntArray(dims);
  auto* affine_quantization = new TfLiteAffineQuantization;
  affine_quantization->scale = TfLiteFloatArrayCreate(scale_size);
  // Initialize scale values to 1.0f
  for (int i = 0; i < scale_size; ++i) {
    affine_quantization->scale->data[i] = 1.0f;
  }
  affine_quantization->zero_point = TfLiteIntArrayCreate(scale_size);
  // Initialize zero_point values to 0
  for (int i = 0; i < scale_size; ++i) {
    affine_quantization->zero_point->data[i] = 0;
  }
  tensor.quantization.params = affine_quantization;
  tensor.quantization.type = kTfLiteAffineQuantization;
  return tensors_.size() - 1;
}

int StubContextBuilder::AddScalarTensor(TfLiteType dtype, void* scalar_value) {
  TfLiteTensor& tensor = tensors_.emplace_back();
  tensor.data.data = scalar_value;
  tensor.type = dtype;
  const int num_elements = 1;
  tensor.dims = TfLiteIntArrayCreate(num_elements);
  size_t size;
  ::tflite::GetSizeOfType(nullptr, dtype, &size);
  tensor.bytes = size * num_elements;
  tensor.dims->data[0] = 1;
  return tensors_.size() - 1;
}

int StubContextBuilder::AddConstTensor(TfLiteType dtype,
                                       absl::Span<const int> dims) {
  TfLiteTensor& tensor = tensors_.emplace_back();
  std::memset(&tensor, 0, sizeof(TfLiteTensor));
  tensor.type = dtype;
  tensor.dims = CopySpanToTfLiteIntArray(dims);
  tensor.allocation_type = kTfLiteMmapRo;
  return tensors_.size() - 1;
}

int StubContextBuilder::AddScalarConstTensor(TfLiteType dtype,
                                             void* scalar_value) {
  TfLiteTensor& tensor = tensors_.emplace_back();
  std::memset(&tensor, 0, sizeof(TfLiteTensor));
  tensor.data.data = scalar_value;
  tensor.type = dtype;
  const int num_elements = 1;
  tensor.dims = TfLiteIntArrayCreate(num_elements);
  size_t size;
  ::tflite::GetSizeOfType(nullptr, dtype, &size);
  tensor.bytes = size * num_elements;
  tensor.dims->data[0] = 1;
  tensor.allocation_type = kTfLiteMmapRo;
  return tensors_.size() - 1;
}

void StubContextBuilder::SetOp(TfLiteBuiltinOperator op, int version,
                               const void* params, absl::Span<const int> inputs,
                               absl::Span<const int> outputs) {
  g_registration->builtin_code = static_cast<int>(op);
  g_registration->version = version;
  TfLiteIntArrayFree(g_node->inputs);
  g_node->inputs = CopySpanToTfLiteIntArray(inputs);
  TfLiteIntArrayFree(g_node->outputs);
  g_node->outputs = CopySpanToTfLiteIntArray(outputs);
  g_node->builtin_data = const_cast<void*>(params);
  TfLiteIntArrayFree(g_delegate_params->input_tensors);
  g_delegate_params->input_tensors = TfLiteIntArrayCopy(g_node->inputs);
  TfLiteIntArrayFree(g_delegate_params->output_tensors);
  g_delegate_params->output_tensors = TfLiteIntArrayCopy(g_node->outputs);
}

void StubContextBuilder::SetOpCustom(const char* name, int version,
                                     const void* params,
                                     absl::Span<const int> inputs,
                                     absl::Span<const int> outputs) {
  g_registration->builtin_code = kTfLiteBuiltinCustom;
  g_registration->custom_name = name;
  g_registration->version = version;
  TfLiteIntArrayFree(g_node->inputs);
  g_node->inputs = CopySpanToTfLiteIntArray(inputs);
  TfLiteIntArrayFree(g_node->outputs);
  g_node->outputs = CopySpanToTfLiteIntArray(outputs);
  g_node->builtin_data = const_cast<void*>(params);
  TfLiteIntArrayFree(g_delegate_params->input_tensors);
  g_delegate_params->input_tensors = TfLiteIntArrayCopy(g_node->inputs);
  TfLiteIntArrayFree(g_delegate_params->output_tensors);
  g_delegate_params->output_tensors = TfLiteIntArrayCopy(g_node->outputs);
}

void StubContextBuilder::SetOpCustomInitialData(const void* data,
                                                size_t length) {
  g_node->custom_initial_data = const_cast<void*>(data);
  g_node->custom_initial_data_size = length;
}

void StubContextBuilder::SetSparsity(int tensor_idx) {
  TfLiteTensor& tensor = tensors_[tensor_idx];
  tensor.sparsity = new TfLiteSparsity;  // Allocate a dummy sparsity struct
  std::memset(tensor.sparsity, 0, sizeof(TfLiteSparsity));
}

void StubContextBuilder::SetConst(int tensor_idx) {
  TfLiteTensor& tensor = tensors_[tensor_idx];
  tensor.allocation_type = kTfLiteMmapRo;
}

TfLiteContext* StubContextBuilder::Build() {
  if (tensors_.empty()) return nullptr;
  context_.tensors_size = tensors_.size();
  context_.tensors = &tensors_[0];
  return &context_;
}

void LoggingReportError(TfLiteContext*, const char* format, ...) {  // NOLINT
  va_list args;

  // Get buffer size.
  va_start(args, format);
  const int size = vsnprintf(nullptr, 0, format, args) + 1;
  va_end(args);
  if (size < 1) {
    ABSL_LOG(ERROR) << format;  // at least log `format`
    return;
  }

  // Log formatted string.
  auto buffer = std::make_unique<char[]>(size);
  if (!buffer) {
    ABSL_LOG(ERROR) << format;  // at least log `format`
    return;
  }
  va_start(args, format);
  vsnprintf(buffer.get(), size, format, args);
  va_end(args);
  ABSL_LOG(ERROR) << buffer.get();
}

}  // namespace litert::ml_drift::ir
