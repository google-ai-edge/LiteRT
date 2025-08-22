// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_TFL_UTILS_H_
#define ODML_LITERT_LITERT_RUNTIME_TFL_UTILS_H_

#include <cstddef>
#include <functional>

#include "litert/c/litert_layout.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "litert/core/options.h"
#include "litert/runtime/tensor_identifier.h"
#include "tflite/c/c_api_types.h"
#include "tflite/interpreter.h"

namespace litert::internal {

// Binds an external memory buffer to a specific input tensor in the
// interpreter. This function sets the tensor's allocation type to
// kTfLiteCustom, making it appear as a constant tensor with a pre-allocated
// buffer.
TfLiteStatus SetCustomAllocationForInputTensor(
    tflite::Interpreter* interpreter,
    const LiteRtExternalTensorBinding& binding);

Expected<ElementType> ConvertElementType(TfLiteType tfl_type);

Expected<Layout> ConvertTensorLayout(
    const TfLiteOpaqueTensor* tfl_opaque_tensor);

Expected<RankedTensorType> ConvertTensorType(
    const TfLiteOpaqueTensor* tfl_opaque_tensor);

// Resize a given `tfl_opaque_tensor` based on a given `layout`.
Expected<void> ResizeTensor(const LiteRtLayout& layout,
                            TfLiteOpaqueContext* tfl_context,
                            TfLiteOpaqueTensor* tfl_opaque_tensor);

struct TensorIdentifierHash {
  std::size_t operator()(const TfLiteTensorIdentifier& id) const {
    return std::hash<int>()(id.subgraph_idx) ^
           (std::hash<int>()(id.tensor_idx) << 1);
  }
};

struct TensorIdentifierEqual {
  bool operator()(const TfLiteTensorIdentifier& lhs,
                  const TfLiteTensorIdentifier& rhs) const {
    return lhs.subgraph_idx == rhs.subgraph_idx &&
           lhs.tensor_idx == rhs.tensor_idx;
  }
};

// Returns the TfLiteTensorIdentifier for the given tensor, or nullopt if the
// tensor is not found in the interpreter.
litert::Expected<TfLiteTensorIdentifier> GetTensorIdentifier(
    const tflite::Interpreter& interpreter, const TfLiteTensor* target_tensor);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_TFL_UTILS_H_
