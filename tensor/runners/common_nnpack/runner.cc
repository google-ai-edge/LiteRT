/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/common_nnpack/runner.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"

namespace litert::tensor {

namespace common_nnpack::internal {

absl::Status Reserve(std::shared_ptr<Buffer>& buffer, size_t required_bytes,
                     bool preserve_data) {
  if (buffer == nullptr) {
    buffer = OwningCpuBuffer::Allocate<Type::kI8>(required_bytes);
    return absl::OkStatus();
  }

  LockedBufferSpan<const std::byte> lock = buffer->Lock();
  const size_t actual_bytes = lock.size();
  if (actual_bytes >= required_bytes) {
    return absl::OkStatus();
  }

  // Check if buffer is owned and can be safely reallocated.
  if (buffer->IsA(OwningCpuBuffer::TypeId())) {
    auto new_buffer = OwningCpuBuffer::Allocate<Type::kI8>(required_bytes);
    if (preserve_data) {
      std::memcpy(new_buffer->data(), lock.data(), actual_bytes);
    }
    buffer = std::move(new_buffer);
    return absl::OkStatus();
  }

  // Non-owning views (SpanCpuBuffer, MutableSpanCpuBuffer, etc.) cannot be
  // resized.
  return absl::InvalidArgumentError(absl::StrFormat(
      "Buffer for is a non-owning view of size %v bytes, which is "
      "smaller than the required %v bytes and cannot be resized",
      actual_bytes, required_bytes));
}

}  // namespace common_nnpack::internal

}  // namespace litert::tensor
