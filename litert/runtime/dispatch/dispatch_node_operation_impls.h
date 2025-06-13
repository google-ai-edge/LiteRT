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

#ifndef ODML_LITERT_LITERT_RUNTIME_DISPATCH_NODE_OPERATION_IMPLS_H_
#define ODML_LITERT_LITERT_RUNTIME_DISPATCH_NODE_OPERATION_IMPLS_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"
#include "litert/runtime/dispatch/dispatch_node_operations.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_types.h"

namespace litert::internal::node_ops {

// Operation to compute and register buffer requirements
class BufferRequirementsOp : public NodeOperation<BufferRequirementsOp> {
 private:
  DispatchDelegateKernel* kernel_;
  int node_idx_;

 public:
  BufferRequirementsOp(DispatchDelegateKernel* kernel, int node_idx)
      : kernel_(kernel), node_idx_(node_idx) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                  int port_idx, TfLiteOpaqueTensor* tensor) {
    LITERT_ASSIGN_OR_RETURN(
        auto result,
        kernel_->GetBufferRequirements(node_idx_, tensor, port_idx, true));

    LITERT_RETURN_IF_ERROR(kernel_->buffer_context_->RegisterBufferRequirements(
        tensor, std::move(result)));
    return {};
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                   int port_idx, TfLiteOpaqueTensor* tensor) {
    LITERT_ASSIGN_OR_RETURN(
        auto result,
        kernel_->GetBufferRequirements(node_idx_, tensor, port_idx, false));

    LITERT_RETURN_IF_ERROR(kernel_->buffer_context_->RegisterBufferRequirements(
        tensor, std::move(result)));
    return {};
  }
};

// Operation to attach tensor buffers to invocation contexts
class TensorAttachmentOp : public NodeOperation<TensorAttachmentOp> {
 private:
  using TensorBufferMap =
      absl::node_hash_map<TfLiteOpaqueTensor*,
                          DispatchDelegateKernel::TensorInfo>;
  TensorBufferMap& tensor_buffer_infos_;
  LiteRtDispatchInvocationContext invocation_ctx_;

 public:
  TensorAttachmentOp(TensorBufferMap& infos,
                     LiteRtDispatchInvocationContext ctx)
      : tensor_buffer_infos_(infos), invocation_ctx_(ctx) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                  int port_idx, TfLiteOpaqueTensor* tensor) {
    auto it = tensor_buffer_infos_.find(tensor);
    if (it == tensor_buffer_infos_.end()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Buffer info not found for input tensor");
    }

    if (!it->second.attached) {
      LITERT_RETURN_IF_ERROR(LiteRtDispatchAttachInput(
          invocation_ctx_, port_idx, it->second.buffer_handle));
    }
    return {};
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                   int port_idx, TfLiteOpaqueTensor* tensor) {
    auto it = tensor_buffer_infos_.find(tensor);
    if (it == tensor_buffer_infos_.end()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Buffer info not found for output tensor");
    }

    if (!it->second.attached) {
      LITERT_RETURN_IF_ERROR(LiteRtDispatchAttachOutput(
          invocation_ctx_, port_idx, it->second.buffer_handle));
    }
    return {};
  }
};

// Operation to build port connections
class PortConnectionOp : public NodeOperation<PortConnectionOp> {
 private:
  using ConnectionMap =
      absl::flat_hash_map<TfLiteOpaqueTensor*,
                          std::vector<DispatchDelegateKernel::PortConnection>>;
  ConnectionMap& connections_;
  int node_idx_;

 public:
  PortConnectionOp(ConnectionMap& conn, int node_idx)
      : connections_(conn), node_idx_(node_idx) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                  int port_idx, TfLiteOpaqueTensor* tensor) {
    connections_[tensor].push_back(
        {.node_idx = node_idx_, .port_idx = port_idx, .is_input_port = true});
    return {};
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                   int port_idx, TfLiteOpaqueTensor* tensor) {
    connections_[tensor].push_back(
        {.node_idx = node_idx_, .port_idx = port_idx, .is_input_port = false});
    return {};
  }
};

// Operation to handle async event attachment
class AsyncEventAttachOp : public NodeOperation<AsyncEventAttachOp> {
 private:
  using TensorBufferMap =
      absl::node_hash_map<TfLiteOpaqueTensor*,
                          DispatchDelegateKernel::TensorInfo>;
  TensorBufferMap& tensor_buffer_infos_;
  LiteRtDispatchInvocationContext invocation_ctx_;
  std::vector<LiteRtEvent>& output_events_;

 public:
  AsyncEventAttachOp(TensorBufferMap& infos,
                     LiteRtDispatchInvocationContext ctx,
                     std::vector<LiteRtEvent>& events)
      : tensor_buffer_infos_(infos),
        invocation_ctx_(ctx),
        output_events_(events) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                  int port_idx, TfLiteOpaqueTensor* tensor) {
    auto it = tensor_buffer_infos_.find(tensor);
    if (it != tensor_buffer_infos_.end() &&
        it->second.tensor_buffer.HasEvent()) {
      auto event_result = it->second.tensor_buffer.GetEvent();
      if (!event_result) return event_result.Error();

      LITERT_RETURN_IF_ERROR(LiteRtDispatchAttachInputEvent(
          invocation_ctx_, port_idx, event_result->Get()));
    }
    return {};
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                   int port_idx, TfLiteOpaqueTensor* tensor) {
    auto it = tensor_buffer_infos_.find(tensor);
    if (it != tensor_buffer_infos_.end() &&
        port_idx < static_cast<int>(output_events_.size())) {
      it->second.tensor_buffer.SetEvent(
          Event(output_events_[port_idx], OwnHandle::kYes));
    }
    return {};
  }
};

// Validation operation for debugging
class ValidationOp : public NodeOperation<ValidationOp> {
 private:
  const char* stage_name_;

 public:
  explicit ValidationOp(const char* name) : stage_name_(name) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                  int index, TfLiteOpaqueTensor* tensor) {
    if (!tensor) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        absl::StrFormat("%s: Invalid input tensor at index %d",
                                        stage_name_, index));
    }
    return {};
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext*, TfLiteOpaqueNode*,
                                   int index, TfLiteOpaqueTensor* tensor) {
    if (!tensor) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        absl::StrFormat("%s: Invalid output tensor at index %d",
                                        stage_name_, index));
    }
    return {};
  }
};

}  // namespace litert::internal::node_ops

#endif  // ODML_LITERT_LITERT_RUNTIME_DISPATCH_NODE_OPERATION_IMPLS_H_
