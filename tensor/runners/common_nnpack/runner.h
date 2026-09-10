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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMMON_NNPACK_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMMON_NNPACK_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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

class ExternalBuffer {
 public:
  absl::Span<std::byte> data() {
    return IsOwned()
               ? absl::MakeSpan(owned_buffer_.data(), owned_buffer_.size())
               : external_view_;
  }
  absl::Span<const std::byte> data() const {
    return IsOwned()
               ? absl::MakeSpan(owned_buffer_.data(), owned_buffer_.size())
               : external_view_;
  }

  void SetExternalView(absl::Span<const std::byte> data) {
    external_view_ =
        absl::MakeSpan(const_cast<std::byte*>(data.data()), data.size());
    owned_buffer_.clear();
  }

  void SetOwnedBuffer(absl::Span<const std::byte> data) {
    owned_buffer_.assign(data.begin(), data.end());
    external_view_ = {};
  }

  absl::Status Resize(size_t new_size) {
    if (IsOwned()) {
      owned_buffer_.resize(new_size);
    } else if (new_size > external_view_.size()) {
      owned_buffer_.resize(new_size);
      std::memcpy(owned_buffer_.data(), external_view_.data(),
                  external_view_.size());
      external_view_ = {};
    }
    return absl::OkStatus();
  }

  bool IsOwned() const { return external_view_.empty(); }

 private:
  std::vector<std::byte> owned_buffer_;
  absl::Span<std::byte> external_view_;
};

inline size_t ByteSize(const graph::TensorInformation& info) {
  return BufferSize(info.type, info.GetSize());
}

class NnpackRunner {
 public:
  explicit NnpackRunner(std::unique_ptr<NnpackGraph> graph)
      : graph_(std::move(graph)) {}

  virtual ~NnpackRunner() = default;

  NnpackRunner(NnpackRunner&&) = default;
  NnpackRunner& operator=(NnpackRunner&&) = default;
  NnpackRunner(const NnpackRunner&) = delete;
  NnpackRunner& operator=(const NnpackRunner&) = delete;

  virtual void SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }

  absl::Status SetInput(const TensorHandle& tensor,
                        const TensorHandle& external_tensor) {
    LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
    NnpackValue& value = graph_->mutable_values()[index];
    if ((value.flags & FlagExternalInput()) == 0) {
      return absl::InvalidArgumentError(
          "Tensor is not marked as external input");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& external_info,
                                graph::GetInfo(external_tensor.GetRaw()));
    if (external_info.type != value.info.type) {
      return absl::InvalidArgumentError(
          absl::StrFormat("External tensor type mismatch: expected %d, got %d",
                          static_cast<int>(value.info.type),
                          static_cast<int>(external_info.type)));
    }
    value.info.shape = external_info.shape;
    ExternalBuffer& held_buffer = external_buffers_[value.id];
    LRT_TENSOR_RETURN_IF_ERROR(held_buffer.Resize(ByteSize(value.info)));
    LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, external_tensor.GetBuffer());
    LockedBufferSpan<const std::byte> lock = buffer.Lock();
    held_buffer.SetExternalView(absl::MakeSpan(lock));
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data,
                        bool copy_data = false) {
    LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
    NnpackValue& value = graph_->mutable_values()[index];
    if ((value.flags & FlagExternalInput()) == 0) {
      return absl::InvalidArgumentError(
          "Tensor is not marked as external input");
    }
    if (ByteSize(value.info) != data.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Mismatched input size: expected %x, got %v",
                          ByteSize(value.info), data.size()));
    }
    if (!copy_data) {
      external_buffers_[value.id].SetExternalView(data);
    } else {
      external_buffers_[value.id].SetOwnedBuffer(data);
    }
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor, absl::Span<std::byte> data,
                        bool copy_data = false) {
    return SetInput(tensor, absl::Span<const std::byte>(data), copy_data);
  }

  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInput(const TensorHandle& tensor,
                        const ContiguousSequence& seq) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The sequence type doesn't match the input tensor type.");
    }
    return SetInput(tensor, absl::Span<const std::byte>(
                                reinterpret_cast<const std::byte*>(seq.data()),
                                seq.size() * sizeof(T)));
  }

  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInput(const TensorHandle& tensor,
                        const ContiguousSequence&& seq) = delete;

  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInputAsCopy(const TensorHandle& tensor,
                              ContiguousSequence&& seq) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The sequence type doesn't match the input tensor type.");
    }
    return SetInput(tensor,
                    absl::Span<const std::byte>(
                        reinterpret_cast<const std::byte*>(seq.data()),
                        seq.size() * sizeof(T)),
                    /*copy_data=*/true);
  }

  absl::Status SetOutput(const TensorHandle& tensor,
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
    external_buffers_[value.id].SetExternalView(data);
    return absl::OkStatus();
  }

  absl::Status ReshapeInput(const TensorHandle& tensor,
                            absl::Span<const int32_t> shape) {
    LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
    NnpackValue& value = graph_->mutable_values()[index];
    if ((value.flags & FlagExternalInput()) == 0) {
      return absl::InvalidArgumentError(
          "Tensor is not marked as external input");
    }
    value.info.shape.assign(shape.begin(), shape.end());
    return external_buffers_[value.id].Resize(ByteSize(value.info));
  }

  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          absl::Span<const std::byte> data) {
    LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
    NnpackValue& value = graph_->mutable_values()[index];
    if ((value.flags & FlagExternalInput()) == 0) {
      return absl::InvalidArgumentError(
          "Tensor is not marked as external input");
    }
    absl::Span<std::byte> buffer = external_buffers_[value.id].data();
    if (offset_bytes + data.size() > buffer.size()) {
      return absl::InvalidArgumentError(
          "Data to write exceeds the external buffer size");
    }
    std::memcpy(buffer.data() + offset_bytes, data.data(), data.size());
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          absl::Span<const T> data) {
    return WriteInput(tensor, offset_bytes,
                      absl::Span<const std::byte>(
                          reinterpret_cast<const std::byte*>(data.data()),
                          data.size() * sizeof(T)));
  }

  template <typename Sequence>
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          const Sequence& seq) {
    using T = typename Sequence::value_type;
    return WriteInput(tensor, offset_bytes,
                      absl::Span<const std::byte>(
                          reinterpret_cast<const std::byte*>(seq.data()),
                          seq.size() * sizeof(T)));
  }

  absl::Status PrepareRuntime() {
    if (!runtime_prepared_) {
      LRT_TENSOR_RETURN_IF_ERROR(CreateRuntime(num_threads_));
      runtime_prepared_ = true;
    }
    return absl::OkStatus();
  }

  absl::Status Run() {
    LRT_TENSOR_RETURN_IF_ERROR(PrepareRuntime());

    // 1. Reshape inputs
    for (auto& value : graph_->mutable_values()) {
      if ((value.flags & FlagExternalInput()) == 0) {
        continue;
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(const std::vector<size_t> dims,
                                  ToNnpackDims(value.info.shape));
      LRT_TENSOR_RETURN_IF_ERROR(SetExternalValueShape(value.id, dims));
      LRT_TENSOR_RETURN_IF_ERROR(
          external_buffers_[value.id].Resize(ByteSize(value.info)));
    }

    // 2. Reshape runtime
    LRT_TENSOR_RETURN_IF_ERROR(ReshapeRuntime());

    // 3. Resize outputs
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
      LRT_TENSOR_RETURN_IF_ERROR(
          external_buffers_[value.id].Resize(ByteSize(value.info)));
    }

    // 4. Set external value data
    LRT_TENSOR_RETURN_IF_ERROR(SetupExternalValues(
        absl::MakeSpan(graph_->mutable_values()), external_buffers_));

    // 5. Invoke runtime
    return InvokeRuntime();
  }

  absl::StatusOr<LockedBufferSpan<const std::byte>> ReadOutput(
      const TensorHandle& tensor) const {
    LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
    const auto& value = graph_->values()[index];
    if ((value.flags & FlagExternalOutput()) == 0) {
      return absl::InvalidArgumentError("Tensor is not marked as output");
    }
    const auto buffer_it = external_buffers_.find(value.id);
    if (buffer_it == external_buffers_.end()) {
      return absl::FailedPreconditionError("Output buffer not found");
    }
    const auto buffer = buffer_it->second.data();
    return LockedBufferSpan<const std::byte>(
        buffer.data(), [](const std::byte*) {}, buffer.size());
  }

  template <class T>
  absl::StatusOr<LockedBufferSpan<const T>> ReadOutputAs(
      const TensorHandle& tensor) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The read type doesn't match the output tensor type.");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(LockedBufferSpan<const std::byte> out,
                                ReadOutput(tensor));
    return std::move(out).template As<const T>();
  }

  const NnpackGraph& graph() const { return *graph_; }
  NnpackGraph& mutable_graph() { return *graph_; }
  const absl::flat_hash_map<uint32_t, ExternalBuffer>& external_buffers()
      const {
    return external_buffers_;
  }
  absl::flat_hash_map<uint32_t, ExternalBuffer>& mutable_external_buffers() {
    return external_buffers_;
  }

 protected:
  virtual uint32_t FlagExternalInput() const = 0;
  virtual uint32_t FlagExternalOutput() const = 0;

  virtual absl::Status CreateRuntime(size_t num_threads) = 0;
  virtual absl::Status SetExternalValueShape(uint32_t id,
                                             absl::Span<const size_t> dims) = 0;
  virtual absl::Status ReshapeRuntime() = 0;
  virtual absl::Status GetExternalValueShape(uint32_t id,
                                             std::vector<size_t>& dims) = 0;
  virtual absl::Status SetupExternalValues(
      absl::Span<NnpackValue> values,
      absl::flat_hash_map<uint32_t, ExternalBuffer>& external_buffers) = 0;
  virtual absl::Status InvokeRuntime() = 0;

  std::unique_ptr<NnpackGraph> graph_;
  absl::flat_hash_map<uint32_t, ExternalBuffer> external_buffers_;
  size_t num_threads_ = 1;
  bool runtime_prepared_ = false;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMMON_NNPACK_RUNNER_H_
