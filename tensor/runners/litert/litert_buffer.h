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
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opencl_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "tensor/buffer.h"
#include "tensor/internal/type_id.h"

namespace litert::tensor {

// Represents a hardware-accelerated tensor buffer in the LiteRT Tensor API.
// It directly wraps a litert::TensorBuffer, providing zero-copy execution
// across LiteRT runners, asynchronous sync via litert::Event, and
// multi-hardware support (AHWB, OpenCL, dmabuf, Metal, WebGPU, FastRPC, and
// aligned host memory).
class LitertBuffer : public Buffer {
 public:
  explicit LitertBuffer(litert::TensorBuffer tensor_buffer,
                        std::shared_ptr<litert::Environment> env = nullptr)
      : env_(std::move(env)), tensor_buffer_(std::move(tensor_buffer)) {}

  ~LitertBuffer() override = default;

  // --- Factory Methods ---

  // Allocates a managed hardware tensor buffer with shared Environment
  // lifetime.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManaged(
      std::shared_ptr<litert::Environment> env,
      litert::TensorBufferType buffer_type,
      const litert::RankedTensorType& tensor_type, size_t buffer_size) {
    if (!env) {
      return absl::InvalidArgumentError("Environment must not be null");
    }
    auto tb_or = litert::TensorBuffer::CreateManaged(*env, buffer_type,
                                                     tensor_type, buffer_size);
    if (!tb_or) {
      return absl::InternalError(tb_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*tb_or), std::move(env));
  }

  // Allocates a managed hardware tensor buffer of the given type and size.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManaged(
      const litert::Environment& env, litert::TensorBufferType buffer_type,
      const litert::RankedTensorType& tensor_type, size_t buffer_size) {
    auto tb_or = litert::TensorBuffer::CreateManaged(env, buffer_type,
                                                     tensor_type, buffer_size);
    if (!tb_or) {
      return absl::InternalError(tb_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*tb_or));
  }

  // Allocates a managed hardware tensor buffer satisfying requirements.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManaged(
      const litert::Environment& env,
      const litert::RankedTensorType& tensor_type,
      const litert::TensorBufferRequirements& requirements) {
    auto tb_or = litert::TensorBuffer::CreateManagedFromRequirements(
        env, tensor_type, requirements);
    if (!tb_or) {
      return absl::InternalError(tb_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*tb_or));
  }

  // Allocates a 64-byte aligned host memory buffer with shared Environment.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManagedHost(
      std::shared_ptr<litert::Environment> env,
      const litert::RankedTensorType& tensor_type, size_t buffer_size) {
    return CreateManaged(std::move(env),
                         litert::TensorBufferType::kHostMemory, tensor_type,
                         buffer_size);
  }

  // Allocates a 64-byte aligned host memory buffer with shared Environment
  // from shape.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManagedHost(
      std::shared_ptr<litert::Environment> env,
      absl::Span<const int32_t> shape, litert::ElementType element_type,
      size_t buffer_size) {
    litert::Layout layout(litert::Dimensions(shape.begin(), shape.end()));
    litert::RankedTensorType tensor_type(element_type, std::move(layout));
    return CreateManagedHost(std::move(env), tensor_type, buffer_size);
  }

  // Allocates a 64-byte aligned host memory buffer with delegates' padding.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManagedHost(
      const litert::Environment& env,
      const litert::RankedTensorType& tensor_type, size_t buffer_size) {
    return CreateManaged(env, litert::TensorBufferType::kHostMemory,
                         tensor_type, buffer_size);
  }

  // Convenience overload allocating host memory from shape, element type,
  // and size.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateManagedHost(
      const litert::Environment& env, absl::Span<const int32_t> shape,
      litert::ElementType element_type, size_t buffer_size) {
    litert::Layout layout(litert::Dimensions(shape.begin(), shape.end()));
    litert::RankedTensorType tensor_type(element_type, std::move(layout));
    return CreateManagedHost(env, tensor_type, buffer_size);
  }

  // Wraps an existing host memory buffer without transferring ownership.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateFromHostMemory(
      const litert::Environment& env,
      const litert::RankedTensorType& tensor_type, void* host_mem_addr,
      size_t buffer_size) {
    auto tb_or = litert::TensorBuffer::CreateFromHostMemory(
        env, tensor_type, host_mem_addr, buffer_size);
    if (!tb_or) {
      return absl::InternalError(tb_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*tb_or));
  }

#if LITERT_HAS_AHWB_SUPPORT
  // Wraps an existing Android AHardwareBuffer without transferring ownership.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateFromAhwb(
      const litert::Environment& env,
      const litert::RankedTensorType& tensor_type, AHardwareBuffer* ahwb,
      size_t ahwb_offset = 0) {
    auto tb_or = litert::TensorBuffer::CreateFromAhwb(env, tensor_type, ahwb,
                                                     ahwb_offset);
    if (!tb_or) {
      return absl::InternalError(tb_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*tb_or));
  }
#endif

#if LITERT_HAS_OPENCL_SUPPORT
  // Wraps an existing OpenCL memory object (e.g. from ML Drift) without copies.
  static absl::StatusOr<std::shared_ptr<LitertBuffer>> CreateFromClBuffer(
      const litert::Environment& env,
      const litert::RankedTensorType& tensor_type,
      litert::TensorBufferType buffer_type, LiteRtClMem cl_memory,
      size_t size_bytes) {
    auto tb_or = litert::TensorBuffer::CreateFromClBuffer(
        env, tensor_type, buffer_type, cl_memory, size_bytes);
    if (!tb_or) {
      return absl::InternalError(tb_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*tb_or));
  }
#endif

  // Duplicates the buffer handle with reference counting.
  absl::StatusOr<std::shared_ptr<LitertBuffer>> Duplicate() const {
    auto dup_or = tensor_buffer_.Duplicate();
    if (!dup_or) {
      return absl::InternalError(dup_or.Error().Message());
    }
    return std::make_shared<LitertBuffer>(std::move(*dup_or), env_);
  }

  // --- Synchronization Fences ---

  bool HasEvent() const { return tensor_buffer_.HasEvent(); }

  absl::StatusOr<litert::Event> GetEvent() const {
    auto event_or = tensor_buffer_.GetEvent();
    if (!event_or) {
      return absl::InternalError(event_or.Error().Message());
    }
    return std::move(*event_or);
  }

  absl::Status SetEvent(litert::Event&& event) {
    auto res = tensor_buffer_.SetEvent(std::move(event));
    if (!res) {
      return absl::InternalError(res.Error().Message());
    }
    return absl::OkStatus();
  }

  absl::Status ClearEvent() {
    auto res = tensor_buffer_.ClearEvent();
    if (!res) {
      return absl::InternalError(res.Error().Message());
    }
    return absl::OkStatus();
  }

  // --- Accessors & Properties ---

  const litert::TensorBuffer& tensor_buffer() const { return tensor_buffer_; }
  litert::TensorBuffer& tensor_buffer() { return tensor_buffer_; }
  const std::shared_ptr<litert::Environment>& env() const { return env_; }

  absl::StatusOr<litert::TensorBufferType> BufferType() const {
    auto type_or = tensor_buffer_.BufferType();
    if (!type_or) return absl::InternalError(type_or.Error().Message());
    return *type_or;
  }

  absl::StatusOr<size_t> Size() const {
    auto size_or = tensor_buffer_.Size();
    if (!size_or) return absl::InternalError(size_or.Error().Message());
    return *size_or;
  }

  absl::StatusOr<size_t> PackedSize() const {
    auto size_or = tensor_buffer_.PackedSize();
    if (!size_or) return absl::InternalError(size_or.Error().Message());
    return *size_or;
  }

  absl::StatusOr<litert::RankedTensorType> TensorType() const {
    auto type_or = tensor_buffer_.TensorType();
    if (!type_or) return absl::InternalError(type_or.Error().Message());
    return *type_or;
  }

  // --- Buffer Interface Implementation ---

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
    auto env = env_;

    auto addr_or = shared_tb->Lock(litert::TensorBuffer::LockMode::kRead);
    ABSL_CHECK(addr_or.HasValue());
    auto size_or = shared_tb->PackedSize();
    ABSL_CHECK(size_or.HasValue());

    return LockedBufferSpan<const std::byte>(
        reinterpret_cast<const std::byte*>(*addr_or),
        [env, shared_tb](const std::byte*) {
          auto status = shared_tb->Unlock();
          ABSL_CHECK(status.HasValue());
        },
        *size_or);
  }

  LockedBufferSpan<std::byte> LockMutable() override {
    auto dup_or = tensor_buffer_.Duplicate();
    ABSL_CHECK(dup_or.HasValue());
    auto shared_tb = std::make_shared<litert::TensorBuffer>(std::move(*dup_or));
    auto env = env_;

    auto addr_or = shared_tb->Lock(litert::TensorBuffer::LockMode::kReadWrite);
    ABSL_CHECK(addr_or.HasValue());
    auto size_or = shared_tb->PackedSize();
    ABSL_CHECK(size_or.HasValue());

    return LockedBufferSpan<std::byte>(
        reinterpret_cast<std::byte*>(*addr_or),
        [env, shared_tb](std::byte*) {
          auto status = shared_tb->Unlock();
          ABSL_CHECK(status.HasValue());
        },
        *size_or);
  }

 private:
  std::shared_ptr<litert::Environment> env_;
  litert::TensorBuffer tensor_buffer_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_BUFFER_H_
