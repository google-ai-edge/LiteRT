/* Copyright 2025 Google LLC.

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

#include "tensor/runners/xnnpack/runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "pthreadpool.h"  // from @pthreadpool

namespace litert::tensor {
namespace {

size_t ByteSize(const graph::TensorInformation& info) {
  return BufferSize(info.type, info.GetSize());
}

absl::StatusOr<std::vector<size_t>> ToXnnDims(
    const std::vector<int32_t>& shape) {
  std::vector<size_t> dims;
  dims.reserve(shape.size());
  for (const int32_t d : shape) {
    if (d < 0) {
      return absl::InvalidArgumentError("Negative dimension is not supported");
    }
    dims.push_back(static_cast<size_t>(d));
  }
  return dims;
}

}  // namespace

// Creates an XnnpackRunner from a list of output tensors.
absl::StatusOr<XnnpackRunner> XnnpackRunner::Create(
    std::vector<TensorHandle> outputs) {
  // Build the XNNPACK graph from the output tensors.
  LRT_TENSOR_ASSIGN_OR_RETURN(std::unique_ptr<XnnpackGraph> graph,
                              BuildXnnpackGraph(std::move(outputs)));
  return XnnpackRunner(std::move(graph));
}

XnnpackRunner::XnnpackRunner(std::unique_ptr<XnnpackGraph> graph)
    : graph_(std::move(graph)) {}

// Sets the input data for a given tensor.
absl::Status XnnpackRunner::SetInput(const TensorHandle& tensor,
                                     absl::Span<const std::byte> data,
                                     const bool copy_data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  // Check if the tensor is marked as an external input.
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  if (ByteSize(value.info) != data.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched input size: expected ", ByteSize(value.info),
                     ", got ", data.size()));
  }
  if (!copy_data) {
    external_buffers_[value.id].SetExternalView(data);
  } else {
    external_buffers_[value.id].SetOwnedBuffer(data);
  }
  return absl::OkStatus();
}

absl::Status XnnpackRunner::SetOutput(const TensorHandle& tensor,
                                      absl::Span<std::byte> data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as output");
  }
  if (ByteSize(value.info) != data.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched output size: expected ", ByteSize(value.info),
                     ", got ", data.size()));
  }
  external_buffers_[value.id].SetExternalView(data);
  return absl::OkStatus();
}

absl::Status XnnpackRunner::ReshapeInput(const TensorHandle& tensor,
                                         absl::Span<const int32_t> shape) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  value.info.shape.assign(shape.begin(), shape.end());
  return external_buffers_[value.id].Resize(ByteSize(value.info));
}

absl::Status XnnpackRunner::WriteInput(const TensorHandle& tensor,
                                       size_t offset_bytes,
                                       absl::Span<const std::byte> data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  auto buffer = external_buffers_[value.id].data();
  if (offset_bytes > buffer.size() ||
      data.size() > buffer.size() - offset_bytes) {
    return absl::InvalidArgumentError("WriteInput out of bounds");
  }
  absl::c_copy(data, buffer.begin() + offset_bytes);
  return absl::OkStatus();
}

absl::Status XnnpackRunner::PrepareRuntime() {
  if (runtime_ == nullptr) {
    if (num_threads_ > 1) {
      threadpool_.reset(pthreadpool_create(num_threads_));
    }
    xnn_runtime* raw_runtime = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_create_runtime_v3(graph_->subgraph(), weights_cache_,
                              /*threadpool=*/threadpool_.get(),
                              /*flags=*/0, &raw_runtime));
    runtime_.reset(raw_runtime);
  }
  return absl::OkStatus();
}

// Runs the XNNPACK graph.
absl::Status XnnpackRunner::Run() {
  LRT_TENSOR_RETURN_IF_ERROR(PrepareRuntime());

  // Reshape external inputs to match the current host-side tensor shapes.
  for (XnnpackValue& value : graph_->mutable_values()) {
    if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
      continue;
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(const std::vector<size_t> dims,
                                ToXnnDims(value.info.shape));
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_reshape_external_value(runtime_.get(), value.id, dims.size(),
                                   dims.empty() ? nullptr : dims.data()))
        << "xnn_reshape_external_value";

    LRT_TENSOR_RETURN_IF_ERROR(
        external_buffers_[value.id].Resize(ByteSize(value.info)));
  }

  // Reshape the runtime (propagates input shapes through the graph).
  LRT_TENSOR_RETURN_IF_ERROR(xnn_reshape_runtime(runtime_.get()));

  // Resize external output buffers to match the runtime-determined shapes.
  for (XnnpackValue& value : graph_->mutable_values()) {
    if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) == 0) {
      continue;
    }

    size_t num_dims = 0;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> dims{};
    LRT_TENSOR_RETURN_IF_ERROR(xnn_get_external_value_shape(
        runtime_.get(), value.id, &num_dims, dims.data()))
        << "xnn_get_external_value_shape";

    value.info.shape.clear();
    value.info.shape.reserve(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
      value.info.shape.push_back(static_cast<int32_t>(dims[i]));
    }

    LRT_TENSOR_RETURN_IF_ERROR(
        external_buffers_[value.id].Resize(ByteSize(value.info)));
  }

  // Prepare external values for the runtime.
  std::vector<xnn_external_value> externals;
  externals.reserve(graph_->values().size());
  for (XnnpackValue& value : graph_->mutable_values()) {
    if (value.flags == 0) {
      continue;
    }
    auto buffer = external_buffers_[value.id].data();
    if (buffer.data() == nullptr) {
      return absl::FailedPreconditionError(
          "External value missing host buffer");
    }
    externals.push_back({.id = value.id, .data = buffer.data()});
  }

  // Setup the runtime with the external values.
  LRT_TENSOR_RETURN_IF_ERROR(
      xnn_setup_runtime_v2(runtime_.get(), externals.size(), externals.data()));
  // Invoke the runtime.
  return XnnStatusToAbsl(xnn_invoke_runtime(runtime_.get()),
                         "xnn_invoke_runtime");
}

absl::StatusOr<LockedBufferSpan<const std::byte>> XnnpackRunner::ReadOutput(
    const TensorHandle& tensor) const {
  // Find the index of the tensor in the graph.
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  const XnnpackValue& value = graph_->values()[index];
  // Check if the tensor is marked as an external output.
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as output");
  }
  const auto buffer_it = external_buffers_.find(value.id);
  if (buffer_it == external_buffers_.end()) {
    return absl::FailedPreconditionError("Tensor is not an output buffer");
  }
  auto buffer = buffer_it->second.data();
  if (buffer.empty()) {
    return absl::FailedPreconditionError("No buffer available for output");
  }
  return LockedBufferSpan<const std::byte>(
      buffer.data(), [](const std::byte*) {}, buffer.size());
}

absl::Span<std::byte> XnnpackRunner::ExternalBuffer::data() {
  return IsOwned() ? absl::MakeSpan(owned_buffer_.data(), owned_buffer_.size())
                   : external_view_;
}

absl::Span<const std::byte> XnnpackRunner::ExternalBuffer::data() const {
  return IsOwned() ? absl::MakeSpan(owned_buffer_.data(), owned_buffer_.size())
                   : external_view_;
}

void XnnpackRunner::ExternalBuffer::SetExternalView(
    absl::Span<const std::byte> data) {
  external_view_ =
      absl::MakeSpan(const_cast<std::byte*>(data.data()), data.size());
  owned_buffer_.clear();
}

void XnnpackRunner::ExternalBuffer::SetOwnedBuffer(
    absl::Span<const std::byte> data) {
  owned_buffer_.assign(data.begin(), data.end());
  external_view_ = {};
}

absl::Status XnnpackRunner::ExternalBuffer::Resize(size_t new_size) {
  if (IsOwned()) {
    owned_buffer_.resize(new_size);
  } else if (new_size > external_view_.size()) {
    // Copy the data of the external view to the resized owned buffer.
    owned_buffer_.resize(new_size);
    memcpy(owned_buffer_.data(), external_view_.data(), external_view_.size());
    external_view_ = {};
  } else {
    external_view_ = absl::MakeSpan(external_view_.data(), new_size);
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor
