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

#include "litert/vendors/apple/bytecode.h"

#include <cstring>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::apple {

namespace {

template <typename T>
void AppendValue(std::vector<uint8_t>& buffer, const T& value) {
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
  buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
}

template <typename T>
void AppendVector(std::vector<uint8_t>& buffer, const std::vector<T>& vec) {
  if (vec.empty()) return;
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(vec.data());
  buffer.insert(buffer.end(), ptr, ptr + vec.size() * sizeof(T));
}

template <typename T>
bool ReadValue(const uint8_t*& ptr, const uint8_t* end, T& value) {
  if (ptr + sizeof(T) > end) return false;
  std::memcpy(&value, ptr, sizeof(T));
  ptr += sizeof(T);
  return true;
}

template <typename T>
bool ReadVector(const uint8_t*& ptr, const uint8_t* end, size_t elements,
                std::vector<T>& vec) {
  size_t bytes = elements * sizeof(T);
  if (ptr + bytes > end) return false;
  vec.resize(elements);
  std::memcpy(vec.data(), ptr, bytes);
  ptr += bytes;
  return true;
}

}  // namespace

Expected<std::vector<uint8_t>> PackMlxBytecode(const MlxBytecode& bytecode) {
  std::vector<uint8_t> buffer;
  // Reserve some space to avoid too many reallocations
  buffer.reserve(1024 + bytecode.weights_data.size() +
                 bytecode.bias_data.size());

  // 1. Magic and Version
  AppendValue(buffer, kAppleMlxMagic);
  AppendValue(buffer, kAppleMlxBytecodeVersion);

  // 2. Weights
  uint32_t weights_rank = bytecode.weights_dims.size();
  AppendValue(buffer, weights_rank);
  AppendVector(buffer, bytecode.weights_dims);
  uint32_t weights_type_val = static_cast<uint32_t>(bytecode.weights_type);
  AppendValue(buffer, weights_type_val);
  uint64_t weights_size = bytecode.weights_data.size();
  AppendValue(buffer, weights_size);
  AppendVector(buffer, bytecode.weights_data);

  // 3. Bias
  uint8_t has_bias_val = bytecode.has_bias ? 1 : 0;
  AppendValue(buffer, has_bias_val);
  if (bytecode.has_bias) {
    uint32_t bias_rank = bytecode.bias_dims.size();
    AppendValue(buffer, bias_rank);
    AppendVector(buffer, bytecode.bias_dims);
    uint32_t bias_type_val = static_cast<uint32_t>(bytecode.bias_type);
    AppendValue(buffer, bias_type_val);
    uint64_t bias_size = bytecode.bias_data.size();
    AppendValue(buffer, bias_size);
    AppendVector(buffer, bytecode.bias_data);
  }

  // 4. Activation
  AppendValue(buffer, bytecode.activation);

  return buffer;
}

Expected<MlxBytecode> ParseMlxBytecode(const void* data, size_t size) {
  if (data == nullptr || size < 8) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid bytecode data or size");
  }

  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  const uint8_t* end = ptr + size;

  // 1. Magic and Version
  uint32_t magic = 0;
  if (!ReadValue(ptr, end, magic) || magic != kAppleMlxMagic) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid magic number");
  }

  uint32_t version = 0;
  if (!ReadValue(ptr, end, version) || version != kAppleMlxBytecodeVersion) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unsupported bytecode version");
  }

  MlxBytecode bytecode;

  // 2. Weights
  uint32_t weights_rank = 0;
  if (!ReadValue(ptr, end, weights_rank)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read weights rank");
  }
  if (!ReadVector(ptr, end, weights_rank, bytecode.weights_dims)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read weights dimensions");
  }
  uint32_t weights_type_val = 0;
  if (!ReadValue(ptr, end, weights_type_val)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read weights type");
  }
  bytecode.weights_type = static_cast<LiteRtElementType>(weights_type_val);
  uint64_t weights_size = 0;
  if (!ReadValue(ptr, end, weights_size)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read weights size");
  }
  if (!ReadVector(ptr, end, weights_size, bytecode.weights_data)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read weights data");
  }

  // 3. Bias
  uint8_t has_bias_val = 0;
  if (!ReadValue(ptr, end, has_bias_val)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read has_bias flag");
  }
  bytecode.has_bias = (has_bias_val != 0);
  if (bytecode.has_bias) {
    uint32_t bias_rank = 0;
    if (!ReadValue(ptr, end, bias_rank)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Failed to read bias rank");
    }
    if (!ReadVector(ptr, end, bias_rank, bytecode.bias_dims)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Failed to read bias dimensions");
    }
    uint32_t bias_type_val = 0;
    if (!ReadValue(ptr, end, bias_type_val)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Failed to read bias type");
    }
    bytecode.bias_type = static_cast<LiteRtElementType>(bias_type_val);
    uint64_t bias_size = 0;
    if (!ReadValue(ptr, end, bias_size)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Failed to read bias size");
    }
    if (!ReadVector(ptr, end, bias_size, bytecode.bias_data)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Failed to read bias data");
    }
  }

  // 4. Activation
  if (!ReadValue(ptr, end, bytecode.activation)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to read activation");
  }

  if (ptr != end) {
    // We could allow trailing bytes, but for strictness we check it.
    // Actually, in some cases there might be padding, so maybe we don't error.
  }

  return bytecode;
}

}  // namespace litert::apple
