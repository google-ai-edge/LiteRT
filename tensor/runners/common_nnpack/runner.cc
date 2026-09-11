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
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/common_nnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

namespace {

// Ensures that `buffer` can hold `required_bytes`.
//
// If `buffer` doesn't point to a buffer, an owning buffer is created with the
// required size.
//
// If the buffer is an owning buffer and is too small, a new one is allocated
// to replace it and, if `preserve_data` is `true`, the data is copied over.
//
// In other cases, if the buffer is too small, this fails.
absl::Status Reserve(std::shared_ptr<Buffer>& buffer, size_t required_bytes,
                     bool preserve_data) {
  if (buffer == nullptr) {
    buffer = OwningCpuBuffer::Allocate<Type::kI8>(required_bytes);
    return absl::OkStatus();
  }

  // TODO(b/493560478): Make it possible to get a buffer size without locking it
  // as this may be an expensive operation.
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
      "Buffer is a non-owning view of size %v bytes, which is smaller than the "
      "required %v bytes and cannot be resized",
      actual_bytes, required_bytes));
}

size_t ByteSize(const graph::TensorInformation& info) {
  return BufferSize(info.type, info.GetSize());
}

}  // namespace

absl::Status NnpackRunner::SetInput(const TensorHandle& tensor,
                                    const TensorHandle& external_tensor) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  NnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & FlagExternalInput()) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& external_info,
                              graph::GetInfo(external_tensor.GetRaw()));
  if (external_info.type != value.info.type) {
    return absl::InvalidArgumentError(
        absl::StrFormat("External tensor type mismatch: expected %d, got %d",
                        static_cast<int>(value.info.type),
                        static_cast<int>(external_info.type)));
  }
  std::shared_ptr<Buffer> buffer_ptr = external_tensor.GetBufferPtr();
  if (buffer_ptr == nullptr) {
    return absl::InvalidArgumentError(
        "Source tensor doesn't have a buffer attached.");
  }
  LRT_TENSOR_RETURN_IF_ERROR(Reserve(buffer_ptr, ByteSize(external_info),
                                     /*preserve_data=*/true));
  external_buffers_[value.id] = std::move(buffer_ptr);
  value.info.shape = external_info.shape;
  return absl::OkStatus();
}

absl::Status NnpackRunner::SetInput(const TensorHandle& tensor,
                                    absl::Span<const std::byte> data,
                                    bool copy_data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  NnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & FlagExternalInput()) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  if (ByteSize(value.info) != data.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Mismatched input size: expected %v, got %v",
                        ByteSize(value.info), data.size()));
  }
  if (copy_data) {
    external_buffers_[value.id] = OwningCpuBuffer::Copy(
        reinterpret_cast<const char*>(data.data()), data.size());
  } else {
    external_buffers_[value.id] =
        std::make_shared<SpanCpuBuffer>(data.data(), data.size());
  }
  return absl::OkStatus();
}

absl::Status NnpackRunner::SetOutput(const TensorHandle& tensor,
                                     absl::Span<std::byte> data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  NnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & FlagExternalOutput()) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as output");
  }
  if (ByteSize(value.info) != data.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Mismatched output size: expected %v, got %v",
                        ByteSize(value.info), data.size()));
  }
  external_buffers_[value.id] =
      std::make_shared<MutableSpanCpuBuffer>(data.data(), data.size());
  return absl::OkStatus();
}

absl::Status NnpackRunner::ReshapeInput(const TensorHandle& tensor,
                                        absl::Span<const int32_t> shape) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  NnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & FlagExternalInput()) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  value.info.shape.assign(shape.begin(), shape.end());

  if (auto it = external_buffers_.find(value.id);
      it != external_buffers_.end()) {
    if (it->second == nullptr || it->second->IsA(OwningCpuBuffer::TypeId())) {
      return Reserve(it->second, ByteSize(value.info),
                     /*preserve_data=*/false);
    }
  }
  return absl::OkStatus();
}

absl::Status NnpackRunner::WriteInput(const TensorHandle& tensor,
                                      size_t offset_bytes,
                                      absl::Span<const std::byte> data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  NnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & FlagExternalInput()) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  auto it = external_buffers_.find(value.id);
  if (it == external_buffers_.end() || it->second == nullptr) {
    return absl::FailedPreconditionError("Input buffer not found");
  }
  auto lock = it->second->LockMutable();
  if (lock.data() == nullptr) {
    return absl::InvalidArgumentError("Input buffer is not mutable");
  }
  if (offset_bytes + data.size() > lock.size()) {
    return absl::InvalidArgumentError(
        "Data to write exceeds the external buffer size");
  }
  std::memcpy(lock.data() + offset_bytes, data.data(), data.size());
  return absl::OkStatus();
}

absl::Status NnpackRunner::PrepareRuntime() {
  if (!runtime_prepared_) {
    LRT_TENSOR_RETURN_IF_ERROR(CreateRuntime(num_threads_));
    runtime_prepared_ = true;
  }
  return absl::OkStatus();
}

absl::Status NnpackRunner::Run() {
  LRT_TENSOR_RETURN_IF_ERROR(PrepareRuntime());

  // Reshape inputs.
  for (auto& value : graph_->mutable_values()) {
    if ((value.flags & FlagExternalInput()) == 0) {
      continue;
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(const std::vector<size_t> dims,
                                ToNnpackDims(value.info.shape));
    LRT_TENSOR_RETURN_IF_ERROR(SetExternalValueShape(value.id, dims));
    std::shared_ptr<Buffer> input_buffer = external_buffers_[value.id];
    if (input_buffer == nullptr) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Value %v (tensor '%s') doesn't have associated buffer.", value.id,
          value.info.name));
    }
    LRT_TENSOR_RETURN_IF_ERROR(Reserve(input_buffer, ByteSize(value.info),
                                       /*preserve_data=*/true));
  }

  // Reshape runtime.
  LRT_TENSOR_RETURN_IF_ERROR(ReshapeRuntime());

  // Resize outputs.
  for (auto& value : graph_->mutable_values()) {
    if ((value.flags & FlagExternalOutput()) == 0) {
      continue;
    }
    std::vector<size_t> dims;
    LRT_TENSOR_RETURN_IF_ERROR(GetExternalValueShape(value.id, dims));

    value.info.shape.clear();
    value.info.shape.reserve(dims.size());
    for (size_t dim : dims) {
      value.info.shape.push_back(static_cast<int32_t>(dim));
    }
    LRT_TENSOR_RETURN_IF_ERROR(Reserve(external_buffers_[value.id],
                                       ByteSize(value.info),
                                       /*preserve_data=*/false));
  }

  // Set external value data and keep locks active during InvokeRuntime.
  std::vector<LockedBufferSpan<const std::byte>> locks;
  LRT_TENSOR_RETURN_IF_ERROR(SetupExternalValues(
      absl::MakeSpan(graph_->mutable_values()), external_buffers_, locks));

  // Invoke runtime
  return InvokeRuntime();
}

absl::StatusOr<LockedBufferSpan<const std::byte>> NnpackRunner::ReadOutput(
    const TensorHandle& tensor) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  const auto& value = graph_->values()[index];
  if ((value.flags & FlagExternalOutput()) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as output");
  }
  const auto buffer_it = external_buffers_.find(value.id);
  if (buffer_it == external_buffers_.end() || buffer_it->second == nullptr) {
    return absl::FailedPreconditionError("Output buffer not found");
  }
  return buffer_it->second->Lock();
}

}  // namespace litert::tensor
