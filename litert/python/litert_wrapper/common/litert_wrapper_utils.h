// Copyright 2025 Google LLC.
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

#ifndef LITERT_PYTHON_LITERT_WRAPPER_COMMON_LITERT_WRAPPER_UTILS_H_
#define LITERT_PYTHON_LITERT_WRAPPER_COMMON_LITERT_WRAPPER_UTILS_H_

#include <Python.h>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"

namespace litert::litert_wrapper_utils {

// The name used for LiteRtTensorBuffer capsules
constexpr absl::string_view kLiteRtTensorBufferName = "LiteRtTensorBuffer";

// Safely destroys a LiteRtTensorBuffer from a PyCapsule and clears the name
// to prevent double destruction. Also releases the model reference stored in
// the capsule context (if any) to ensure correct destruction order.
void DestroyTensorBufferFromCapsule(PyObject* capsule);

// Creates a PyCapsule for a TensorBuffer with the appropriate destructor.
// If model_wrapper is provided, stores a reference to it in the capsule
// context. This ensures the model stays alive as long as any of its buffers
// exist, preventing use-after-free crashes during garbage collection.
PyObject* MakeTensorBufferCapsule(TensorBuffer& buffer,
                                  PyObject* model_wrapper);

}  // namespace litert::litert_wrapper_utils

#endif  // LITERT_PYTHON_LITERT_WRAPPER_COMMON_LITERT_WRAPPER_UTILS_H_
