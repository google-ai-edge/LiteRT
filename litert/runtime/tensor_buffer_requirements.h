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

#ifndef ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REQUIREMENTS_H_
#define ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REQUIREMENTS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"

class LiteRtTensorBufferRequirementsT {
 public:
  LiteRtTensorBufferRequirementsT(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, std::vector<uint32_t>&& strides,
      size_t alignment = LITERT_HOST_MEMORY_BUFFER_ALIGNMENT)
      : supported_buffer_types_(
            supported_tensor_buffer_types,
            supported_tensor_buffer_types + num_supported_tensor_buffer_types),
        buffer_size_(buffer_size),
        strides_(std::move(strides)),
        alignment_(alignment) {}
  const std::vector<LiteRtTensorBufferType>& SupportedBufferTypes() const {
    return supported_buffer_types_;
  }
  size_t BufferSize() const { return buffer_size_; }
  const std::vector<uint32_t>& Strides() const { return strides_; }
  size_t Alignment() const { return alignment_; }
  std::string ToString() const;

 private:
  friend litert::Expected<std::unique_ptr<LiteRtTensorBufferRequirementsT>>
  JoinInternal(const LiteRtTensorBufferRequirementsT& src1,
               const LiteRtTensorBufferRequirementsT& src2);

  std::vector<LiteRtTensorBufferType> supported_buffer_types_;
  size_t buffer_size_;
  // Stride per each dimension.
  std::vector<uint32_t> strides_;
  // Memory alignment requirement in bytes.
  size_t alignment_;
};

namespace litert::internal {

litert::Expected<std::unique_ptr<LiteRtTensorBufferRequirementsT>> Join(
    const LiteRtTensorBufferRequirementsT& src1,
    const LiteRtTensorBufferRequirementsT& src2);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REQUIREMENTS_H_
