/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_KERNELS_FUZZING_FUZZING_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_FUZZING_FUZZING_UTIL_H_

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/core/c/common.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace fuzzing {

enum class RunResult { kSuccess, kRejected, kHarnessFailure };

constexpr size_t kModelBufferAlignment = 16;

class SilentErrorReporter final : public ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    (void)format;
    (void)args;
    return 0;
  }
};

inline bool CheckedMultiply(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr) {
    return false;
  }
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

inline bool CheckedAdd(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr ||
      lhs > std::numeric_limits<size_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

inline bool CheckedAddInt64(int64_t lhs, int64_t rhs, int64_t* result) {
  if (result == nullptr) {
    return false;
  }
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

inline bool CheckedSubInt64(int64_t lhs, int64_t rhs, int64_t* result) {
  if (result == nullptr) {
    return false;
  }
  if ((rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs) ||
      (rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs)) {
    return false;
  }
  *result = lhs - rhs;
  return true;
}

inline bool CheckedMulInt64(int64_t lhs, int64_t rhs, int64_t* result) {
  if (result == nullptr) {
    return false;
  }
  if (lhs == 0 || rhs == 0) {
    *result = 0;
    return true;
  }
  if (lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) {
    return false;
  }
  if (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()) {
    return false;
  }
  if (lhs > 0) {
    if (rhs > 0) {
      if (lhs > std::numeric_limits<int64_t>::max() / rhs) return false;
    } else if (rhs < std::numeric_limits<int64_t>::min() / lhs) {
      return false;
    }
  } else {
    if (rhs > 0) {
      if (lhs < std::numeric_limits<int64_t>::min() / rhs) return false;
    } else if (lhs < std::numeric_limits<int64_t>::max() / rhs) {
      return false;
    }
  }
  *result = lhs * rhs;
  return true;
}

inline bool CheckedShapeElementCount(const std::vector<int32_t>& shape,
                                     size_t* result) {
  if (result == nullptr) {
    return false;
  }
  size_t count = 1;
  for (const int32_t dim : shape) {
    if (dim < 0 ||
        !CheckedMultiply(count, static_cast<size_t>(dim), &count)) {
      return false;
    }
  }
  *result = count;
  return true;
}

inline size_t TypeSize(TensorType type) {
  switch (type) {
    case TensorType_FLOAT32:
      return sizeof(float);
    case TensorType_UINT8:
      return sizeof(uint8_t);
    case TensorType_INT8:
      return sizeof(int8_t);
    case TensorType_INT4:
      return sizeof(int8_t);
    case TensorType_INT16:
      return sizeof(int16_t);
    case TensorType_INT32:
      return sizeof(int32_t);
    case TensorType_INT64:
      return sizeof(int64_t);
    case TensorType_BOOL:
      return sizeof(bool);
    default:
      return 0;
  }
}

inline bool StorageBytesForElements(TensorType type, size_t count,
                                    size_t* bytes) {
  if (bytes == nullptr) {
    return false;
  }
  if (type == TensorType_INT4) {
    *bytes = (count + 1) / 2;
    return true;
  }
  return CheckedMultiply(count, TypeSize(type), bytes);
}

inline size_t TypeAlignment(TensorType type) {
  switch (type) {
    case TensorType_FLOAT32:
      return alignof(float);
    case TensorType_UINT8:
      return alignof(uint8_t);
    case TensorType_INT8:
      return alignof(int8_t);
    case TensorType_INT4:
      return alignof(uint8_t);
    case TensorType_INT16:
      return alignof(int16_t);
    case TensorType_INT32:
      return alignof(int32_t);
    case TensorType_INT64:
      return alignof(int64_t);
    case TensorType_BOOL:
      return alignof(bool);
    default:
      return 0;
  }
}

inline size_t TypeAlignment(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return alignof(float);
    case kTfLiteUInt8:
      return alignof(uint8_t);
    case kTfLiteInt8:
      return alignof(int8_t);
    case kTfLiteInt4:
      return alignof(uint8_t);
    case kTfLiteInt16:
      return alignof(int16_t);
    case kTfLiteInt32:
      return alignof(int32_t);
    case kTfLiteInt64:
      return alignof(int64_t);
    case kTfLiteBool:
      return alignof(bool);
    default:
      return 0;
  }
}

inline bool IsAligned(const void* data, size_t alignment) {
  return alignment != 0 &&
         reinterpret_cast<uintptr_t>(data) % alignment == 0;
}

inline flatbuffers::Offset<Buffer> CreateAlignedBuffer(
    flatbuffers::FlatBufferBuilder* builder,
    const std::vector<uint8_t>& bytes) {
  return CreateBufferDirect(*builder, &bytes);
}

inline bool TensorBufferDataIsAligned(const Tensor* tensor,
                                      const Buffer* buffer) {
  if (tensor == nullptr || buffer == nullptr || buffer->data() == nullptr ||
      buffer->data()->size() == 0) {
    return true;
  }
  const void* data = buffer->data()->data();
  return IsAligned(data, kModelBufferAlignment) &&
         IsAligned(data, TypeAlignment(tensor->type()));
}

inline bool ConstantTensorBuffersAreAligned(const Model* model) {
  if (model == nullptr || model->buffers() == nullptr ||
      model->subgraphs() == nullptr) {
    return false;
  }
  for (const SubGraph* subgraph : *model->subgraphs()) {
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
      return false;
    }
    for (const Tensor* tensor : *subgraph->tensors()) {
      if (tensor == nullptr || tensor->buffer() == 0 ||
          tensor->buffer() >= model->buffers()->size()) {
        continue;
      }
      if (!TensorBufferDataIsAligned(tensor,
                                     model->buffers()->Get(tensor->buffer()))) {
        return false;
      }
    }
  }
  return true;
}

inline bool TensorDataIsAligned(const TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->data.raw == nullptr || tensor->bytes == 0) {
    return true;
  }
  return IsAligned(tensor->data.raw, TypeAlignment(tensor->type));
}

template <typename T>
void FillValues(std::vector<uint8_t>* bytes, size_t count, int64_t seed) {
  if constexpr (std::is_same_v<T, bool>) {
    std::vector<uint8_t> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = ((seed + static_cast<int64_t>(i % 7)) % 7) != 0;
    }
    *bytes = std::move(values);
  } else {
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = static_cast<T>((seed + static_cast<int64_t>(i % 7)) % 7);
    }
    bytes->resize(values.size() * sizeof(T));
    if (!values.empty()) {
      std::memcpy(bytes->data(), values.data(), bytes->size());
    }
  }
}

inline void FillInt4Values(std::vector<uint8_t>* bytes, size_t count,
                           int64_t seed) {
  bytes->assign((count + 1) / 2, 0);
  for (size_t i = 0; i < count; ++i) {
    const uint8_t nibble =
        static_cast<uint8_t>(static_cast<uint64_t>(seed + i) & 0x0F);
    if ((i & 1) == 0) {
      (*bytes)[i / 2] |= nibble;
    } else {
      (*bytes)[i / 2] |= static_cast<uint8_t>(nibble << 4);
    }
  }
}

inline std::vector<uint8_t> MakeValues(TensorType type, size_t count,
                                       int64_t seed) {
  std::vector<uint8_t> bytes;
  switch (type) {
    case TensorType_FLOAT32:
      FillValues<float>(&bytes, count, seed);
      break;
    case TensorType_UINT8:
      FillValues<uint8_t>(&bytes, count, seed);
      break;
    case TensorType_INT8:
      FillValues<int8_t>(&bytes, count, seed);
      break;
    case TensorType_INT4:
      FillInt4Values(&bytes, count, seed);
      break;
    case TensorType_INT16:
      FillValues<int16_t>(&bytes, count, seed);
      break;
    case TensorType_INT32:
      FillValues<int32_t>(&bytes, count, seed);
      break;
    case TensorType_INT64:
      FillValues<int64_t>(&bytes, count, seed);
      break;
    case TensorType_BOOL:
      FillValues<bool>(&bytes, count, seed);
      break;
    default:
      break;
  }
  return bytes;
}

inline std::vector<uint8_t> MakeIntegerValues(
    TensorType type, const std::vector<int64_t>& values) {
  size_t byte_count = 0;
  if (!StorageBytesForElements(type, values.size(), &byte_count)) {
    return {};
  }
  std::vector<uint8_t> bytes(byte_count, 0);
  auto store = [&bytes](size_t index, auto value) {
    std::memcpy(bytes.data() + index * sizeof(value), &value, sizeof(value));
  };
  for (size_t i = 0; i < values.size(); ++i) {
    switch (type) {
      case TensorType_UINT8:
        store(i, static_cast<uint8_t>(values[i]));
        break;
      case TensorType_INT8:
        store(i, static_cast<int8_t>(values[i]));
        break;
      case TensorType_INT4: {
        const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
        if ((i & 1) == 0) {
          bytes[i / 2] |= nibble;
        } else {
          bytes[i / 2] |= static_cast<uint8_t>(nibble << 4);
        }
        break;
      }
      case TensorType_INT16:
        store(i, static_cast<int16_t>(values[i]));
        break;
      case TensorType_INT32:
        store(i, static_cast<int32_t>(values[i]));
        break;
      case TensorType_INT64:
        store(i, values[i]);
        break;
      case TensorType_BOOL:
        store(i, values[i] != 0);
        break;
      case TensorType_FLOAT32:
        store(i, static_cast<float>(values[i]));
        break;
      default:
        break;
    }
  }
  return bytes;
}

inline std::vector<int64_t> MaterializeValues(
    const std::vector<int64_t>& values, size_t count) {
  std::vector<int64_t> result(count, 0);
  if (values.empty()) return result;
  for (size_t i = 0; i < count; ++i) {
    result[i] = values[i % values.size()];
  }
  return result;
}

inline void OverlayBytes(const std::vector<uint8_t>& overlay,
                         std::vector<uint8_t>* bytes) {
  const size_t count = std::min(overlay.size(), bytes->size());
  if (count != 0) {
    std::memcpy(bytes->data(), overlay.data(), count);
  }
}

inline void ApplyCentralTensorInputInvariants(TensorType type,
                                              std::vector<uint8_t>* bytes) {
  if (type != TensorType_BOOL) {
    return;
  }
  for (uint8_t& byte : *bytes) {
    byte = byte == 0 ? 0 : 1;
  }
}

}  // namespace fuzzing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_FUZZING_FUZZING_UTIL_H_
