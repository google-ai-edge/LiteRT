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

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "litert/c/litert_event.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/common/vendor_dispatch_base.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/southbound.h"

class LiteRtDispatchInvocationContextT
    : public litert::vendors::VendorInvocationContext {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<Ptr> CreateFromBytecode(
      const litert::google_tensor::Southbound& southbound,
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs);

  static litert::Expected<Ptr> CreateFromGraph(
      const litert::google_tensor::Southbound& southbound,
      LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph);

  ~LiteRtDispatchInvocationContextT();

  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> InvokeAsync(int num_output_events,
                                     LiteRtEvent* output_events);
  litert::Expected<void> StartMetricsCollectionInternal(int detail_level);
  litert::Expected<void> StopMetricsCollectionInternal(
      LiteRtDispatchMetrics* metrics);

  // Override base class virtual methods for metrics
  LiteRtStatus StartMetricsCollection(int detail_level) override;
  LiteRtStatus StopMetricsCollection(LiteRtDispatchMetrics* metrics) override;

  litert::Expected<void> AttachInputEvent(int graph_input_index,
                                          LiteRtEvent input_event);

  ThrInvocationContext* thr_invocation_context() {
    return thr_invocation_context_;
  }

  LiteRtDispatchDeviceContext device_context() { return device_context_; }

  LiteRtDispatchGraph graph() { return graph_; }

  // Overrides from VendorInvocationContext
  LiteRtStatus AttachInput(int graph_input_idx,
                           LiteRtTensorBufferHandle handle) override;

  LiteRtStatus AttachOutput(int graph_output_idx,
                            LiteRtTensorBufferHandle handle) override;

  LiteRtStatus DetachInput(int graph_input_idx,
                           LiteRtTensorBufferHandle handle) override;

  LiteRtStatus DetachOutput(int graph_output_idx,
                            LiteRtTensorBufferHandle handle) override;

  LiteRtStatus Invoke() override;

  void* GetBackendContext() override { return thr_invocation_context_; }

 private:
  LiteRtDispatchInvocationContextT(
      const litert::google_tensor::Southbound& southbound,
      ThrInvocationContext* thr_invocation_context,
      LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph)
      : southbound_(southbound),
        thr_invocation_context_(thr_invocation_context),
        device_context_(device_context),
        graph_(graph) {}

  void AttachExecutable(LiteRtDispatchExecutableHandle exec_handle) {
    exec_handle_ = exec_handle;
  }

  const litert::google_tensor::Southbound& southbound_;
  ThrInvocationContext* thr_invocation_context_;
  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchGraph graph_;
  std::optional<LiteRtDispatchExecutableHandle> exec_handle_;
  std::map<std::string, int> input_sync_fences_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
