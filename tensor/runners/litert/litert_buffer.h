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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_BUFFER_H_

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"
#include "tensor/buffer.h"
#include "tensor/internal/type_id.h"

namespace litert::tensor {

// A custom buffer implementation that wraps a LiteRT TensorBuffer.
// This allows passing accelerated buffers (e.g., GPU) between runners
// using the high-level Tensor abstraction.
class LitertBuffer : public Buffer {
 public:
  explicit LitertBuffer(litert::TensorBuffer tensor_buffer)
      : tensor_buffer_(std::move(tensor_buffer)) {}

  ~LitertBuffer() override = default;

  const litert::TensorBuffer& tensor_buffer() const { return tensor_buffer_; }

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<LitertBuffer>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<LitertBuffer>();
  }

  LockedBufferSpan<const std::byte> Lock() override {
    auto dup_or = tensor_buffer_.Duplicate();
    ABSL_CHECK(dup_or.HasValue());
    auto shared_tb = std::make_shared<litert::TensorBuffer>(std::move(*dup_or));

    auto addr_or = shared_tb->Lock(litert::TensorBuffer::LockMode::kRead);
    ABSL_CHECK(addr_or.HasValue());
    auto size_or = shared_tb->PackedSize();
    ABSL_CHECK(size_or.HasValue());

    return LockedBufferSpan<const std::byte>(
        reinterpret_cast<const std::byte*>(*addr_or),
        [shared_tb](const std::byte*) {
          auto status = shared_tb->Unlock();
          ABSL_CHECK(status.HasValue());
        },
        *size_or);
  }

  LockedBufferSpan<std::byte> LockMutable() override {
    auto dup_or = tensor_buffer_.Duplicate();
    ABSL_CHECK(dup_or.HasValue());
    auto shared_tb = std::make_shared<litert::TensorBuffer>(std::move(*dup_or));

    auto addr_or = shared_tb->Lock(litert::TensorBuffer::LockMode::kReadWrite);
    ABSL_CHECK(addr_or.HasValue());
    auto size_or = shared_tb->PackedSize();
    ABSL_CHECK(size_or.HasValue());

    return LockedBufferSpan<std::byte>(
        reinterpret_cast<std::byte*>(*addr_or),
        [shared_tb](std::byte*) {
          auto status = shared_tb->Unlock();
          ABSL_CHECK(status.HasValue());
        },
        *size_or);
  }

 private:
  litert::TensorBuffer tensor_buffer_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_BUFFER_H_
