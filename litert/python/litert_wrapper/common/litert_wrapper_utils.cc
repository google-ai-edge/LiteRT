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

#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"

#include <Python.h>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert::litert_wrapper_utils {

void DestroyTensorBufferFromCapsule(PyObject* capsule) {
  // TODO(b/414622532): Remove this check, using PyCapsule_GetPointer default
  // behavior.
  if (absl::NullSafeStringView(PyCapsule_GetName(capsule)) ==
      kLiteRtTensorBufferName) {
    if (void* ptr =
            PyCapsule_GetPointer(capsule, kLiteRtTensorBufferName.data());
        ptr) {
      LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(ptr));
      PyCapsule_SetName(capsule, "");
    }
  }
}

PyObject* MakeTensorBufferCapsule(TensorBuffer& buffer) {
  return PyCapsule_New(buffer.Release(), kLiteRtTensorBufferName.data(),
                       &DestroyTensorBufferFromCapsule);
}

}  // namespace litert::litert_wrapper_utils
