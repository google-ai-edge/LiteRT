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
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_event_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_config.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_utils.h"
#if LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
#include "litert/c/options/google/litert_google_tensor_privileged_options_type.h"
#endif  // LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace gt = litert::google_tensor;

namespace {

// Custom LiteRT event backed by a vendor-preferred fence file descriptor.
class GoogleTensorVendorPreferredCustomEvent : public LiteRtCustomEventT {
 public:
  GoogleTensorVendorPreferredCustomEvent(int fd, bool owns_fd)
      : fd_(fd), owns_fd_(owns_fd) {
    Retain = &GoogleTensorVendorPreferredCustomEvent::RetainImpl;
    Release = &GoogleTensorVendorPreferredCustomEvent::ReleaseImpl;
    Wait = &GoogleTensorVendorPreferredCustomEvent::WaitImpl;
    IsSignaled = &GoogleTensorVendorPreferredCustomEvent::IsSignaledImpl;
    GetNative = &GoogleTensorVendorPreferredCustomEvent::GetNativeImpl;
  }

  ~GoogleTensorVendorPreferredCustomEvent() {
    if (owns_fd_ && fd_ >= 0) {
      close(fd_);
    }
  }

 private:
  static void RetainImpl(LiteRtCustomEvent event) {
    if (event == nullptr) return;
    auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
    self->ref_count_.fetch_add(1, std::memory_order_relaxed);
  }

  static void ReleaseImpl(LiteRtCustomEvent event) {
    if (event == nullptr) return;
    auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
    if (self->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete self;
    }
  }

  static void WaitImpl(LiteRtCustomEvent event, int64_t timeout_in_ms) {
    if (event == nullptr) return;
    auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
    if (self->fd_ < 0) {
      LITERT_LOG(LITERT_ERROR, "Invalid vendor-preferred fence fd %d in Wait",
                 self->fd_);
      return;
    }

    pollfd fds = {
        .fd = self->fd_,
        .events = POLLIN,
    };

    int ret;
    do {
      ret = poll(&fds, 1, timeout_in_ms);
      if (ret == 1) {
        if ((fds.revents & (POLLERR | POLLNVAL)) != 0) {
          LITERT_LOG(LITERT_ERROR,
                     "poll returned error revents=0x%x on vendor-preferred "
                     "fence fd %d",
                     fds.revents, self->fd_);
        }
        return;
      } else if (ret == 0) {
        LITERT_LOG(LITERT_WARNING, "Timeout expired: %" PRId64 " ms on fd %d",
                   timeout_in_ms, self->fd_);
        return;
      }
    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

    LITERT_LOG(LITERT_ERROR,
               "Failed to wait on vendor-preferred fence fd %d: %s", self->fd_,
               strerror(errno));
  }

  static int IsSignaledImpl(LiteRtCustomEvent event) {
    if (event == nullptr) return 0;
    auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
    if (self->fd_ < 0) return 0;

    pollfd fds = {
        .fd = self->fd_,
        .events = POLLIN,
    };

    int ret;
    do {
      ret = poll(&fds, 1, /*timeout=*/0);
      if (ret == 1) {
        return (fds.revents & (POLLERR | POLLNVAL)) == 0 ? 1 : 0;
      }
      if (ret == 0) {
        return 0;
      }
    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

    return 0;
  }

  static void* GetNativeImpl(LiteRtCustomEvent event) {
    if (event == nullptr) return nullptr;
    auto* self = static_cast<GoogleTensorVendorPreferredCustomEvent*>(event);
    return static_cast<void*>(&self->fd_);
  }

  std::atomic<int> ref_count_{1};
  int fd_ = -1;
  bool owns_fd_ = false;
};

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

    // On Android platforms, it is expected that `sync_fence_fd` is a dma-fence
    // fd. On other Linux platforms, it is expected that `sync_fence_fd` is an
    // eventfd.
    thr_fence_type =
#if defined(__ANDROID__)
        kThrFenceTypeDma;
#else
        kThrFenceTypeEventFd;
#endif
  } else if (type == LiteRtEventTypeCustom) {
    LiteRtCustomEvent custom_event = nullptr;
    LITERT_RETURN_IF_ERROR(device_context_->runtime_context()->get_custom_event(
        input_event, &custom_event));
    if (custom_event == nullptr || custom_event->GetNative == nullptr) {
      LITERT_LOG(LITERT_ERROR,
                 "Custom input event is null or missing GetNative callback");
      return kLiteRtStatusErrorInvalidArgument;
    }
    void* native = custom_event->GetNative(custom_event);
    if (native == nullptr) {
      LITERT_LOG(LITERT_ERROR,
                 "Custom input event GetNative returned null pointer");
      return kLiteRtStatusErrorInvalidArgument;
    }
    sync_fence_fd = *static_cast<const int*>(native);
    if (sync_fence_fd < 0) {
      LITERT_LOG(LITERT_ERROR, "Invalid custom event fence fd %d",
                 sync_fence_fd);
      return kLiteRtStatusErrorInvalidArgument;
    }
    thr_fence_type = kThrFenceTypeVendorPreferred;
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

bool LiteRtDispatchInvocationContextT::ShouldUseVendorPreferredFence() const {
#if LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  auto dev_priv_opts = device_context_->GetGoogleTensorPrivilegedOptions();
  if (dev_priv_opts.has_value() && dev_priv_opts->use_vendor_preferred_fence) {
    return true;
  }
#endif  // LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  auto dev_gt_opts = device_context_->GetGoogleTensorOptions();
  if (dev_gt_opts.has_value() && dev_gt_opts->use_vendor_preferred_fence) {
    return true;
  }

  return false;
}

LiteRtStatus LiteRtDispatchInvocationContextT::InvokeAsync(
    absl::Span<LiteRtEvent> output_events) {
  if (output_events.size() != graph_->NumOutputEdges()) {
    LITERT_LOG(LITERT_ERROR,
                "Graph has %zu outputs but %zu output events were provided",
                graph_->NumOutputEdges(), output_events.size());
    return kLiteRtStatusErrorInvalidArgument;
  }

  const bool use_vendor_preferred_fence = ShouldUseVendorPreferredFence();
  const ThrFenceType out_fence_type =
      use_vendor_preferred_fence ? kThrFenceTypeVendorPreferred
                                 : kThrFenceTypeDma;

  GT_LOG_RETURN_IF_SB_ERROR(
      thrInvocationContextPrepareForInvoke2(thr_invocation_context_,
                                            out_fence_type),
      "Failed to prepare SB invocation context with out-fence type %d",
      out_fence_type);

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
      LITERT_RETURN_IF_ERROR(
          device_context_->runtime_context()->create_managed_event(
              /*env=*/nullptr, LiteRtEventTypeCustom,
              &output_events[graph_output_index]));
      auto* custom_event = new GoogleTensorVendorPreferredCustomEvent(
          sync_fence_fd, /*owns_fd=*/true);
      std::move(sync_fence_fd_cleanup).Cancel();
      LiteRtStatus set_status =
          device_context_->runtime_context()->set_custom_event(
              output_events[graph_output_index], custom_event);
      custom_event->Release(custom_event);
      LITERT_RETURN_IF_ERROR(set_status);
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
