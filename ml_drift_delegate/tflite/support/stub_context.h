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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_STUB_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_STUB_CONTEXT_H_

#include <cstring>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

// Builds a stub TfLiteContext for unit testing.
//
// SetOp & Build can only be called once. Attempting to call it again will cause
// a segmentation fault.  TODO: who/impjdi - Fix this.
//
//   StubContextBuilder builder;
//   int a = builder.AddTensor(dtype, dims);
//   int b = builder.AddTensor(dtype, dims);
//   int c = builder.AddTensor(dtype, dims);
//   builder.SetOp(kTfLiteBuiltinAdd, /*version=*/1, /*params=*/nullptr,
//                 /*inputs=*/{a, b}, /*outputs=*/{c});
//   TfLiteContext* stub_context = builder.Build();
//   ASSERT_TRUE(stub_context != nullptr);
//
// Minimal implementation with only a single node for now; to be extended on an
// as-needed basis.
class StubContextBuilder {
 public:
  StubContextBuilder();
  ~StubContextBuilder();

  // Adds a TfLiteTensor and returns its index in TfLiteContext.tensors.
  int AddTensor(TfLiteType dtype, absl::Span<const int> dims);

  // Adds a AffineQuantized TfLiteTensor and returns its index in
  // TfLiteContext.tensors.
  int AddQuantizedTensor(TfLiteType dtype, absl::Span<const int> dims,
                         int scale_size = 1);

  // Adds a scalar TfLiteTensor and returns its index in TfLiteContext.tensors.
  int AddScalarTensor(TfLiteType dtype, void* scalar_value);

  // Adds a const TfLiteTensor and returns its index in TfLiteContext.tensors.
  int AddConstTensor(TfLiteType dtype, absl::Span<const int> dims);

  // Adds a 1D const TfLiteTensor from vector data.
  template <typename T>
  int AddConst1dTensor(TfLiteType dtype, absl::Span<const T> data) {
    TfLiteTensor& tensor = tensors_.emplace_back();
    std::memset(&tensor, 0, sizeof(TfLiteTensor));
    tensor.type = dtype;
    tensor.dims = TfLiteIntArrayCreate(1);
    tensor.dims->data[0] = data.size();
    tensor.allocation_type = kTfLiteMmapRo;
    // For testing, we assume the vector's lifetime exceeds the test's duration.
    tensor.data.data = const_cast<T*>(data.data());
    return tensors_.size() - 1;
  }

  // Adds a scalar const TfLiteTensor and returns its index in
  // TfLiteContext.tensors.
  int AddScalarConstTensor(TfLiteType dtype, void* scalar_value);

  // Sets op code, version, params, inputs, and outputs of the node.
  // If params is allocated in the heap, the caller must free it.
  void SetOp(TfLiteBuiltinOperator op, int version, const void* params,
             absl::Span<const int> inputs, absl::Span<const int> outputs);

  // Sets custom op name, version, params, inputs, and outputs of the node.
  void SetOpCustom(const char* name, int version, const void* params,
                   absl::Span<const int> inputs, absl::Span<const int> outputs);

  // Sets custom op initial data.
  void SetOpCustomInitialData(const void* data, size_t length);

  // Sets sparsity for the tensor at `tensor_idx`.
  void SetSparsity(int tensor_idx);

  // Sets the tensor at `tensor_idx` as constant.
  void SetConst(int tensor_idx);

  // Returns the stub TfLiteContext.
  TfLiteContext* Build();

 private:
  TfLiteContext context_;
  std::vector<TfLiteTensor> tensors_;
};

// Suppresses all error reporting for testing.
inline void NullReportError(TfLiteContext*, const char* format, ...) {}

// Not used in the final tests, but handy for debugging.
void LoggingReportError(TfLiteContext*, const char* format, ...);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_STUB_CONTEXT_H_
