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

#ifndef ODML_LITERT_LITERT_CC_LITERT_STRING_UTIL_H_
#define ODML_LITERT_LITERT_CC_LITERT_STRING_UTIL_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace litert::util {

/// @brief Serializes a vector of strings into a single contiguous byte buffer
/// using the TFLite string tensor serialization format.
inline std::vector<uint8_t> SerializeStrings(
    const std::vector<std::string>& strs) {
  std::vector<uint8_t> serialized;
  int32_t num_strs = strs.size();
  size_t total_size = sizeof(int32_t) + sizeof(int32_t) * (num_strs + 1);
  for (const auto& s : strs) {
    total_size += s.size();
  }
  serialized.resize(total_size);
  std::memcpy(serialized.data(), &num_strs, sizeof(int32_t));
  int32_t* offsets =
      reinterpret_cast<int32_t*>(serialized.data() + sizeof(int32_t));
  int32_t current_offset = sizeof(int32_t) + sizeof(int32_t) * (num_strs + 1);
  for (int i = 0; i < num_strs; ++i) {
    offsets[i] = current_offset;
    std::memcpy(serialized.data() + current_offset, strs[i].data(),
                strs[i].size());
    current_offset += strs[i].size();
  }
  offsets[num_strs] = current_offset;
  return serialized;
}

/// @brief Deserializes a contiguous byte buffer into a vector of strings
/// using the TFLite string tensor serialization format.
inline std::vector<std::string> DeserializeStrings(const uint8_t* data,
                                                   size_t size) {
  std::vector<std::string> strs;
  if (size < sizeof(int32_t)) return strs;
  int32_t num_strs;
  std::memcpy(&num_strs, data, sizeof(int32_t));
  if (size < sizeof(int32_t) + sizeof(int32_t) * (num_strs + 1)) return strs;
  const int32_t* offsets =
      reinterpret_cast<const int32_t*>(data + sizeof(int32_t));
  strs.reserve(num_strs);
  for (int i = 0; i < num_strs; ++i) {
    int32_t start = offsets[i];
    int32_t end = offsets[i + 1];
    if (start < 0 || end < start || size < end) {
      return {};
    }
    strs.push_back(
        std::string(reinterpret_cast<const char*>(data + start), end - start));
  }
  return strs;
}

/// @brief Creates a managed TensorBuffer containing serialized strings with an
/// explicit shape.
inline Expected<TensorBuffer> CreateTensorBufferFromStrings(
    const Environment& env, const RankedTensorType& tensor_type,
    const std::vector<std::string>& strs,
    TensorBufferType buffer_type = TensorBufferType::kHostMemory) {
  std::vector<uint8_t> serialized = SerializeStrings(strs);
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer buffer,
      TensorBuffer::CreateManaged(env, buffer_type, tensor_type,
                                  serialized.size()));
  LITERT_RETURN_IF_ERROR(
      buffer.Write<uint8_t>(absl::MakeConstSpan(serialized)));
  return buffer;
}

/// @brief Convenience overload that automatically creates a 1D TensorBuffer.
inline Expected<TensorBuffer> CreateTensorBufferFromStrings(
    const Environment& env, const std::vector<std::string>& strs,
    TensorBufferType buffer_type = TensorBufferType::kHostMemory) {
  auto ranked_type =
      RankedTensorType(ElementType::TfString,
                       Layout(Dimensions({static_cast<int32_t>(strs.size())})));
  return CreateTensorBufferFromStrings(env, ranked_type, strs, buffer_type);
}

/// @brief Reads and deserializes strings from a TensorBuffer.
inline Expected<std::vector<std::string>> GetStringsFromTensorBuffer(
    TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                          TensorBufferScopedLock::Create<const uint8_t>(
                              buffer, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(size_t size, buffer.Size());
  return DeserializeStrings(lock_and_addr.second, size);
}

}  // namespace litert::util

#endif  // ODML_LITERT_LITERT_CC_LITERT_STRING_UTIL_H_
