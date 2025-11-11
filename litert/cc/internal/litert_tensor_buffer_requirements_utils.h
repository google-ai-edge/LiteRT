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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TENSOR_BUFFER_REQUIREMENTS_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TENSOR_BUFFER_REQUIREMENTS_UTILS_H_

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"

namespace litert {
namespace internal {

// Converts a C LiteRtTensorBufferRequirements to a C++
// TensorBufferRequirements.
Expected<TensorBufferRequirements> ToTensorBufferRequirements(
    LiteRtTensorBufferRequirements requirements);

// Converts a C++ TensorBufferRequirements to a C
// LiteRtTensorBufferRequirements.
Expected<LiteRtTensorBufferRequirements> ToLiteRtTensorBufferRequirements(
    const TensorBufferRequirements& requirements);

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TENSOR_BUFFER_REQUIREMENTS_UTILS_H_
