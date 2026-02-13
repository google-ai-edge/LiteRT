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

#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// This class is thread-compatible.
class LiteRtDispatchInvocationContextT {
 public:
  static LiteRtStatus CreateFromBytecode(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer& exec_bytecode_buffer,
      const char* absl_nullable function_name, int num_inputs, int num_outputs,
      LiteRtDispatchInvocationContext& invocation_context);

  static LiteRtStatus CreateFromGraph(
      LiteRtDispatchDeviceContext device_context,
      std::optional<LiteRtDispatchExecutableHandle> exec_handle,
      LiteRtDispatchGraph graph,
      LiteRtDispatchInvocationContext& invocation_context);

  LiteRtStatus Destroy();

  LiteRtStatus AttachInput(int graph_input_index,
                           LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus AttachOutput(int graph_output_index,
                            LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus DetachInput(int graph_input_index,
                           LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus DetachOutput(int graph_output_index,
                            LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus Invoke();

  LiteRtStatus AttachInputEvent(int graph_input_index, LiteRtEvent input_event);

  LiteRtStatus InvokeAsync(absl::Span<LiteRtEvent> output_events);

  LiteRtStatus StartMetricsCollection(int detail_level);

  LiteRtStatus StopMetricsCollection(LiteRtDispatchMetrics& metrics);

  LiteRtStatus SetRunOptions(LiteRtOptions options) {
    run_options_ = options;
    return kLiteRtStatusOk;
  }

  LiteRtOptions GetRunOptions() const { return run_options_; }

  LiteRtStatus SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info == nullptr) {
      scheduling_info_.reset();
      return kLiteRtStatusOk;
    }
    scheduling_info_ = *scheduling_info;
    return kLiteRtStatusOk;
  }

  const LiteRtSchedulingInfo* GetSchedulingInfo() const {
    return scheduling_info_.has_value() ? &(*scheduling_info_) : nullptr;
  }

  ThrInvocationContext* absl_nonnull thr_invocation_context() {
    return thr_invocation_context_;
  }

  LiteRtDispatchGraph graph() { return graph_; }

 private:
  LiteRtDispatchInvocationContextT(
      LiteRtDispatchDeviceContext device_context,
      std::optional<LiteRtDispatchExecutableHandle> exec_handle,
      LiteRtDispatchGraph graph,
      ThrInvocationContext* absl_nonnull thr_invocation_context)
      : device_context_(device_context),
        exec_handle_(exec_handle),
        graph_(graph),
        thr_invocation_context_(thr_invocation_context) {}

  // Consumers of this class must use `Destroy` to delete the instance.
  ~LiteRtDispatchInvocationContextT() = default;

  LiteRtStatus DetachAndUnregisterInFences();

  LiteRtDispatchDeviceContext device_context_;
  // When `exec_handle_` contains a valid value, this means the invocation
  // context was created from bytecode, and therefore is responsible for
  // managing the lifetime of `graph_`.
  std::optional<LiteRtDispatchExecutableHandle> exec_handle_;
  LiteRtDispatchGraph graph_;
  ThrInvocationContext* absl_nonnull thr_invocation_context_;
  // Set to `true` after the invocation context is (optionally) successfully
  // registered with its graph. This prevents 'Destroy' from attempting to
  // unregister the invocation context from its graph when the invocation
  // context has not previously been registered.
  bool registered_with_graph_ = false;
  // Associates an input edge ID with its attached fence.
  absl::flat_hash_map<LiteRtDispatchEdgeId, ThrFenceHandle> in_fences_;

  LiteRtOptions run_options_ = nullptr;

  std::optional<LiteRtSchedulingInfo> scheduling_info_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
