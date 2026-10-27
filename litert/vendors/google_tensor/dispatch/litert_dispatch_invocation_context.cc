// Copyright 2024 Google LLC.
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

#include "litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"

#include <errno.h>
#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_config.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_utils.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace gt = litert::google_tensor;

namespace {

// On Android platforms, a sync fence fd is expected to be a dma-fence fd. On
// other Linux platforms, it is expected to be an eventfd.
constexpr ThrFenceType DefaultInputSyncFenceType() {
#if defined(__ANDROID__)
  return kThrFenceTypeDma;
#else
  return kThrFenceTypeEventFd;
#endif
}

// On Android platforms, a custom event fence fd is expected to be a
// vendor-preferred (IIF) fence fd. On other Linux platforms, output fences are
// backed by eventfds, so input custom fences are registered as eventfds.
constexpr ThrFenceType DefaultInputCustomFenceType() {
#if defined(__ANDROID__)
  return kThrFenceTypeVendorPreferred;
#else
  return kThrFenceTypeEventFd;
#endif
}

int PollFenceFd(int fd, int64_t timeout_in_ms, int16_t* revents = nullptr) {
  pollfd fds = {.fd = fd, .events = POLLIN};
  int ret;
  do {
    ret = poll(&fds, 1, timeout_in_ms);
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
  if (revents != nullptr) {
    *revents = fds.revents;
  }
  return ret;
}

// Custom LiteRT event backed by an owned vendor-preferred fence fd.
class GoogleTensorVendorPreferredCustomEvent : public LiteRtCustomEventT {
 public:
  explicit GoogleTensorVendorPreferredCustomEvent(int fd) : fd_(fd) {
    Retain = [](LiteRtCustomEvent event) {
      if (event != nullptr) {
        static_cast<GoogleTensorVendorPreferredCustomEvent*>(event)
            ->ref_count_.fetch_add(1, std::memory_order_relaxed);
      }
    };
    Release = [](LiteRtCustomEvent event) {
      auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
      if (self != nullptr &&
          self->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete self;
      }
    };
    Wait = [](LiteRtCustomEvent event, int64_t timeout_in_ms) {
      auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
      if (self == nullptr || self->fd_ < 0) return;
      int ret = PollFenceFd(self->fd_, timeout_in_ms);
      if (ret == 0) {
        LITERT_LOG(LITERT_WARNING, "Timeout expired: %" PRId64 " ms on fd %d",
                   timeout_in_ms, self->fd_);
      } else if (ret < 0) {
        LITERT_LOG(LITERT_ERROR,
                   "Failed to wait on vendor-preferred fence fd %d: %s",
                   self->fd_, strerror(errno));
      }
    };
    IsSignaled = [](LiteRtCustomEvent event) -> int {
      auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
      int16_t revents = 0;
      return (self != nullptr && self->fd_ >= 0 &&
              PollFenceFd(self->fd_, /*timeout_in_ms=*/0, &revents) == 1 &&
              (revents & (POLLERR | POLLNVAL)) == 0)
                 ? 1
                 : 0;
    };
    GetNative = [](LiteRtCustomEvent event) -> void* {
      auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
      return self != nullptr ? &self->fd_ : nullptr;
    };
  }

  ~GoogleTensorVendorPreferredCustomEvent() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

 private:
  std::atomic<int> ref_count_{1};
  int fd_ = -1;
};

LiteRtStatus CreateCustomEventFromFd(
    const LiteRtRuntimeContext* runtime_context, int fd,
    LiteRtEvent* out_event) {
  auto* custom_event = new GoogleTensorVendorPreferredCustomEvent(fd);
  absl::Cleanup release_custom_event = [custom_event] {
    custom_event->Release(custom_event);
  };
  LITERT_RETURN_IF_ERROR(runtime_context->create_managed_event(
      /*env=*/nullptr, LiteRtEventTypeCustom, out_event));
  return runtime_context->set_custom_event(*out_event, custom_event);
}

}  // namespace

LiteRtStatus LiteRtDispatchInvocationContextT::CreateFromBytecode(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer& exec_bytecode_buffer,
    const char* absl_nullable function_name, int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext& invocation_context) {
  GT_LOG_RETURN_IF_NULL(device_context);

  LiteRtDispatchExecutableHandle exec_handle;
  LITERT_RETURN_IF_ERROR(device_context->LoadExecutable(
      exec_type, exec_bytecode_buffer, exec_handle));
  absl::Cleanup exec_cleanup = [device_context, exec_handle] {
    device_context->UnloadExecutable(exec_handle);
  };

  LiteRtDispatchGraph graph;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGraphT::Create(device_context, graph));
  absl::Cleanup graph_cleanup = [graph] { graph->Destroy(); };

  LiteRtDispatchNodeId node_id = 0;
  LiteRtDispatchNodeType node_type;
  switch (exec_type) {
    case kLiteRtDispatchExecutableTypeDspLibrary:
      node_type = kLiteRtDispatchNodeTypeDsp;
      break;
    case kLiteRtDispatchExecutableTypeMlModel:
      node_type = kLiteRtDispatchNodeTypeNpu;
      break;
    case kLiteRtDispatchExecutableTypeTfliteModel:
      node_type = kLiteRtDispatchNodeTypeCpu;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Invalid executable type %d", exec_type);
      return kLiteRtStatusErrorInvalidArgument;
  }

  // The positional node connection API is used here so that this path remains
  // compatible with older SouthBound versions. As a result, support for the
  // basic Dispatch API can still be advertised in the presence of an old
  // SouthBound version.
  LITERT_RETURN_IF_ERROR(graph->AddPositionalNode(node_id, node_type));
  LITERT_RETURN_IF_ERROR(
      graph->AssignNodeFunction(node_id, exec_handle, function_name));

  LiteRtDispatchEdgeId next_edge_id = 0;
  for (int input_index = 0; input_index < num_inputs; ++input_index) {
    LiteRtDispatchEdgeId edge_id = next_edge_id++;

    LITERT_RETURN_IF_ERROR(graph->AddEdge(edge_id));
    LITERT_RETURN_IF_ERROR(graph->ConnectPositionalNodeInput(node_id, edge_id));
    LITERT_RETURN_IF_ERROR(graph->ConnectGraphInput(input_index, edge_id));
  }

  for (int output_index = 0; output_index < num_outputs; ++output_index) {
    LiteRtDispatchEdgeId edge_id = next_edge_id++;

    LITERT_RETURN_IF_ERROR(graph->AddEdge(edge_id));
    LITERT_RETURN_IF_ERROR(
        graph->ConnectPositionalNodeOutput(node_id, edge_id));
    LITERT_RETURN_IF_ERROR(graph->ConnectGraphOutput(output_index, edge_id));
  }

  LITERT_RETURN_IF_ERROR(gt::AnnotateCoherencyEdges(
      device_context, function_name, num_inputs, num_outputs, graph));

  LITERT_RETURN_IF_ERROR(
      CreateFromGraph(device_context, exec_handle, graph, invocation_context));

  std::move(graph_cleanup).Cancel();
  std::move(exec_cleanup).Cancel();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::CreateFromGraph(
    LiteRtDispatchDeviceContext device_context,
    std::optional<LiteRtDispatchExecutableHandle> exec_handle,
    LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext& invocation_context) {
  GT_LOG_RETURN_IF_NULL(device_context);
  GT_LOG_RETURN_IF_NULL(graph);

  ThrInvocationContext* thr_invocation_context = thrInvocationContextGet(
      graph->thr_graph(), device_context->thr_context());
  if (thr_invocation_context == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to get SB invocation context");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // The returned instance must be allocated with `new`, as it will be
  // deallocated via `delete` in `Destroy`.
  invocation_context = new LiteRtDispatchInvocationContextT(
      device_context, exec_handle, graph, thr_invocation_context);
  absl::Cleanup invocation_context_cleanup = [invocation_context] {
    invocation_context->Destroy();
  };

  if (!exec_handle.has_value()) {
    // In this case, the invocation context does not own the graph, and thus
    // must be registered with the graph to ensure that the graph remains alive
    // until after the invocation context is destroyed.
    LITERT_RETURN_IF_ERROR(
        graph->RegisterInvocationContext(invocation_context));
    invocation_context->registered_with_graph_ = true;
  }

  std::move(invocation_context_cleanup).Cancel();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::Destroy() {
  LITERT_RETURN_IF_ERROR(DetachAndUnregisterInFences());

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextDelete(graph_->thr_graph(), thr_invocation_context_),
      "Failed to delete SB invocation context");

  if (exec_handle_.has_value()) {
    // In this case, the invocation context owns the graph.
    LITERT_RETURN_IF_ERROR(graph_->Destroy());
    LITERT_RETURN_IF_ERROR(device_context_->UnloadExecutable(*exec_handle_));
  } else if (registered_with_graph_) {
    LITERT_RETURN_IF_ERROR(graph_->UnregisterInvocationContext(this));
  }

  delete this;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LiteRtDispatchEdgeId edge_id;
  LITERT_RETURN_IF_ERROR(graph_->GetInputEdgeId(graph_input_index, edge_id));

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextAttachBuffer(
          thr_invocation_context_, device_context_->thr_context(),
          gt::ToThrEdgeId(edge_id), tensor_buffer_handle),
      "Failed to attach tensor buffer %" PRIu64 " to input edge %" PRIu64,
      tensor_buffer_handle, edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LiteRtDispatchEdgeId edge_id;
  LITERT_RETURN_IF_ERROR(graph_->GetOutputEdgeId(graph_output_index, edge_id));

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextAttachBuffer(
          thr_invocation_context_, device_context_->thr_context(),
          gt::ToThrEdgeId(edge_id), tensor_buffer_handle),
      "Failed to attach tensor buffer %" PRIu64 " to output edge %" PRIu64,
      tensor_buffer_handle, edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LiteRtDispatchEdgeId edge_id;
  LITERT_RETURN_IF_ERROR(graph_->GetInputEdgeId(graph_input_index, edge_id));

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextDetachBuffer(
          thr_invocation_context_, device_context_->thr_context(),
          gt::ToThrEdgeId(edge_id), tensor_buffer_handle),
      "Failed to detach tensor buffer %" PRIu64 " from input edge %" PRIu64,
      tensor_buffer_handle, edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LiteRtDispatchEdgeId edge_id;
  LITERT_RETURN_IF_ERROR(graph_->GetOutputEdgeId(graph_output_index, edge_id));

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextDetachBuffer(
          thr_invocation_context_, device_context_->thr_context(),
          gt::ToThrEdgeId(edge_id), tensor_buffer_handle),
      "Failed to detach tensor buffer %" PRIu64 " from output edge %" PRIu64,
      tensor_buffer_handle, edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::Invoke() {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextPrepareForInvoke(thr_invocation_context_,
                                           /*create_output_sync_fence=*/false),
      "Failed to prepare SB invocation context for invoke");

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextInvokeOnce(thr_invocation_context_),
      "Failed to invoke SB invocation context");

  LITERT_RETURN_IF_ERROR(DetachAndUnregisterInFences());

  GT_LOG_RETURN_IF_SB_ERROR(thrInvocationContextWait(thr_invocation_context_),
                            "Failed to wait for SB invocation context");

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::AttachInputEvent(
    int graph_input_index, LiteRtEvent input_event) {
  LiteRtEventType type;
  LITERT_RETURN_IF_ERROR(
      device_context_->runtime_context()->get_event_event_type(input_event,
                                                               &type));
  int sync_fence_fd = -1;
  ThrFenceType thr_fence_type;
  if (type == LiteRtEventTypeSyncFenceFd) {
    // This API does not return a duped fd, so `sync_fence_fd` must not be
    // closed.
    LITERT_RETURN_IF_ERROR(
        device_context_->runtime_context()->get_event_sync_fence_fd(
            input_event, &sync_fence_fd));

    thr_fence_type = DefaultInputSyncFenceType();
  } else if (type == LiteRtEventTypeCustom) {
    LiteRtCustomEvent custom_event = nullptr;
    LITERT_RETURN_IF_ERROR(device_context_->runtime_context()->get_custom_event(
        input_event, &custom_event));
    GT_LOG_RETURN_IF_NULL(custom_event);
    GT_LOG_RETURN_IF_NULL(custom_event->GetNative);
    void* native = custom_event->GetNative(custom_event);
    GT_LOG_RETURN_IF_NULL(native);
    sync_fence_fd = *static_cast<const int*>(native);
    if (sync_fence_fd < 0) {
      LITERT_LOG(LITERT_ERROR, "Invalid custom event fence fd %d",
                 sync_fence_fd);
      return kLiteRtStatusErrorInvalidArgument;
    }
    thr_fence_type = DefaultInputCustomFenceType();
  } else {
    LITERT_LOG(LITERT_ERROR,
               "Attaching input event with type %d is not supported", type);
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtDispatchEdgeId edge_id;
  LITERT_RETURN_IF_ERROR(graph_->GetInputEdgeId(graph_input_index, edge_id));

  ThrFenceHandle thr_fence_handle;
  GT_LOG_RETURN_IF_SB_ERROR(
      thrRegisterFence(device_context_->thr_context(), thr_fence_type,
                       sync_fence_fd, &thr_fence_handle),
      "Failed to register fence fd %d of type %d with SB", sync_fence_fd,
      thr_fence_type);
  absl::Cleanup thr_fence_handle_cleanup = [this, thr_fence_handle]() {
    thrUnregisterFence(device_context_->thr_context(), thr_fence_handle);
  };

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextAttachInputBufferFence(
          thr_invocation_context_, gt::ToThrEdgeId(edge_id), thr_fence_handle),
      "Failed to attach SB fence %" PRIu64 " to input edge %" PRIu64,
      thr_fence_handle, edge_id);

  in_fences_.insert_or_assign(edge_id, thr_fence_handle);

  std::move(thr_fence_handle_cleanup).Cancel();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::InvokeAsync(
    absl::Span<LiteRtEvent> output_events) {
  if (output_events.size() != graph_->NumOutputEdges()) {
    LITERT_LOG(LITERT_ERROR,
               "Graph has %zu outputs but %zu output events were provided",
               graph_->NumOutputEdges(), output_events.size());
    return kLiteRtStatusErrorInvalidArgument;
  }

  bool use_vendor_preferred_fence = false;
#if LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  if (const auto& options = device_context_->GetGoogleTensorPrivilegedOptions();
      options.has_value()) {
    use_vendor_preferred_fence = options->use_vendor_preferred_fence;
  }
#endif  // LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  if (use_vendor_preferred_fence) {
    if (!vendor_preferred_fence_supported_) {
      use_vendor_preferred_fence = false;
    } else if (thrInvocationContextPrepareForInvoke2(
                   thr_invocation_context_, kThrFenceTypeVendorPreferred) !=
               kThrStatusSuccess) {
      LITERT_LOG(LITERT_WARNING,
                 "Vendor-preferred fence is not supported on this device; "
                 "falling back to DMA fence");
      vendor_preferred_fence_supported_ = false;
      use_vendor_preferred_fence = false;
    }
  }
  if (!use_vendor_preferred_fence) {
    GT_LOG_RETURN_IF_SB_ERROR(
        thrInvocationContextPrepareForInvoke2(thr_invocation_context_,
                                              kThrFenceTypeDma),
        "Failed to prepare SB invocation context with out-fence type %d",
        kThrFenceTypeDma);
  }

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextInvokeOnce(thr_invocation_context_),
      "Failed to invoke SB invocation context");

  LITERT_RETURN_IF_ERROR(DetachAndUnregisterInFences());

  for (int graph_output_index = 0; graph_output_index < output_events.size();
       graph_output_index++) {
    LiteRtDispatchEdgeId edge_id;
    LITERT_RETURN_IF_ERROR(
        graph_->GetOutputEdgeId(graph_output_index, edge_id));

    ThrFenceHandle thr_fence_handle;
    GT_LOG_RETURN_IF_SB_ERROR(thrInvocationContextGetOutputBufferFence(
                                  thr_invocation_context_,
                                  gt::ToThrEdgeId(edge_id), &thr_fence_handle),
                              "Failed to get output fence for edge %" PRIu64,
                              edge_id);
    absl::Cleanup thr_fence_handle_cleanup = [this, thr_fence_handle]() {
      thrUnregisterFence(device_context_->thr_context(), thr_fence_handle);
    };

    int sync_fence_fd;
    // This API returns a duped fd, so `sync_fence_fd` must be closed.
    GT_LOG_RETURN_IF_SB_ERROR(
        thrFenceGetDupFd(device_context_->thr_context(), thr_fence_handle,
                         &sync_fence_fd),
        "Failed to dup fd of SB fence %" PRIu64, thr_fence_handle);
    absl::Cleanup sync_fence_fd_cleanup = [sync_fence_fd]() {
      close(sync_fence_fd);
    };

    if (use_vendor_preferred_fence) {
      std::move(sync_fence_fd_cleanup).Cancel();
      LITERT_RETURN_IF_ERROR(CreateCustomEventFromFd(
          device_context_->runtime_context(), sync_fence_fd,
          &output_events[graph_output_index]));
    } else {
      // `owns_fd=true` so that `sync_fence_fd` is closed when the event is
      // destroyed.
      LITERT_RETURN_IF_ERROR(
          device_context_->runtime_context()->create_event_from_sync_fence_fd(
              /*env=*/nullptr, sync_fence_fd, /*owns_fd=*/true,
              &output_events[graph_output_index]));
      std::move(sync_fence_fd_cleanup).Cancel();
    }

    std::move(thr_fence_handle_cleanup).Cancel();
    GT_LOG_RETURN_IF_SB_ERROR(
        thrUnregisterFence(device_context_->thr_context(), thr_fence_handle),
        "Failed to unregister output SB fence %" PRIu64, thr_fence_handle);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::StartMetricsCollection(
    int detail_level) {
  GT_LOG_RETURN_IF_SB_ERROR(thrInvocationContextStartMetricsCollection(
                                thr_invocation_context_, detail_level),
                            "Failed to start SB metrics collection");

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::StopMetricsCollection(
    LiteRtDispatchMetrics& metrics) {
  ThrInvocationMetrics thr_metrics{
      .version = kThrInvocationMetricsStructVersion,
  };

  GT_LOG_RETURN_IF_SB_ERROR(thrInvocationContextStopMetricsCollection(
                                thr_invocation_context_, &thr_metrics),
                            "Failed to stop SB metrics collection");

  metrics = new LiteRtDispatchMetricsT(thr_metrics);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextT::DetachAndUnregisterInFences() {
  while (!in_fences_.empty()) {
    auto iter = in_fences_.begin();
    LiteRtDispatchEdgeId edge_id = iter->first;
    ThrFenceHandle thr_fence_handle = iter->second;

    GT_LOG_RETURN_IF_SB_ERROR(
        thrInvocationContextDetachInputBufferFence(thr_invocation_context_,
                                                   gt::ToThrEdgeId(edge_id),
                                                   thr_fence_handle),
        "Failed to detach input SB fence %" PRIu64 " from edge %" PRIu64,
        thr_fence_handle, edge_id);

    GT_LOG_RETURN_IF_SB_ERROR(
        thrUnregisterFence(device_context_->thr_context(), thr_fence_handle),
        "Failed to unregister input SB fence %" PRIu64, thr_fence_handle);

    in_fences_.erase(iter);
  }

  return kLiteRtStatusOk;
}
