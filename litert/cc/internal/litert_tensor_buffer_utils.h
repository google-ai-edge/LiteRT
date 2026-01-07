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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TENSOR_BUFFER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TENSOR_BUFFER_UTILS_H_

#include <string>

#include "litert/c/litert_tensor_buffer_types.h"

/// @file
/// @brief Provides utility functions for LiteRT tensor buffers.

namespace litert {

/// @brief Converts a `LiteRtTensorBufferType` enum to its string
/// representation.
/// @param buffer_type The buffer type to convert.
/// @return The string representation of the buffer type.
std::string BufferTypeToString(LiteRtTensorBufferType buffer_type);

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TENSOR_BUFFER_UTILS_H_
