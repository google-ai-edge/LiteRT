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

#ifndef ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_
#define ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/metrics.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils/simple_opaque_delegate.h"

class LiteRtExternalLiteRtBufferContextT;

namespace litert::internal {

// A TFL kernel that the interpreter calls to dispatch execution through the
// Dispatch API.
class DispatchDelegateKernel
    : public tflite::SimpleOpaqueDelegateKernelInterface {
 public:
  using Ptr = std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>;

  ~DispatchDelegateKernel() override;

  static Expected<Ptr> Create(std::string&& graph_name,
                              LiteRtEnvironmentOptions environment_options,
                              LiteRtOptions options,
                              LiteRtDispatchDeviceContext device_context);

  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override;

  Expected<void> StartMetricsCollection(int detail_level);

  Expected<LiteRtMetricsT> StopMetricsCollection();

 private:
  DispatchDelegateKernel(LiteRtEnvironmentOptions environment_options,
                         LiteRtOptions options, std::string&& graph_name,
                         LiteRtDispatchDeviceContext device_context,
                         bool async_dispatch)
      : environment_options_(environment_options),
        options_(options),
        graph_name_(std::move(graph_name)),
        device_context_(std::move(device_context)),
        async_dispatch_(async_dispatch) {}

  static Expected<std::vector<TfLiteOpaqueNode*>> GetNodes(
      TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams& params);
  static Expected<std::vector<TfLiteOpaqueTensor*>> GetTensors(
      const TfLiteOpaqueContext* context, const TfLiteIntArray& tensor_ids);
  static Expected<std::vector<TfLiteOpaqueTensor*>> GetInternalTensors(
      TfLiteOpaqueContext* context, const std::vector<TfLiteOpaqueNode*>& nodes,
      const std::vector<TfLiteOpaqueTensor*>& input_tensors,
      const std::vector<TfLiteOpaqueTensor*>& output_tensors);

  Expected<void> InitHelper(TfLiteOpaqueContext* context,
                            const TfLiteOpaqueDelegateParams& params);
  Expected<void> PrepareHelper(TfLiteOpaqueContext* context,
                               TfLiteOpaqueNode* node);
  Expected<void> EvalHelper(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node);

  Expected<LiteRtDispatchInvocationContext> CreateNodeInvocationContext(
      TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);

  Expected<LiteRtTensorBufferRequirementsPtr> GetBufferRequirements(
      int node_idx, TfLiteOpaqueTensor* io_tfl_tensor, int io_tensor_index,
      bool is_input) const;

  Expected<void> ComputeRequirements(TfLiteOpaqueContext* context);
  Expected<void> ComputeTensorPortConnections(TfLiteOpaqueContext* context);

  Expected<void> AllocateTensorBuffersIfNeeded(TfLiteOpaqueContext* context);
  Expected<LiteRtTensorBufferPtr> AllocateTensorBuffer(
      TfLiteOpaqueTensor* tfl_tensor);
  Expected<void> RegisterBufferWithDispatchApi(
      TfLiteOpaqueContext* context, TfLiteOpaqueTensor* tfl_tensor,
      LiteRtTensorBufferPtr&& tensor_buffer);

  Expected<void> AttachBuffersToInvocationContextsIfNeeded(
      TfLiteOpaqueContext* context);

  Expected<void> ScheduleAsyncExecution(TfLiteOpaqueContext* context);
  Expected<void> ScheduleSyncExecution(TfLiteOpaqueContext* context);

  Expected<const void*> FindAllocBase() const;
  Expected<int> FindAllocBaseFd() const;

  LiteRtEnvironmentOptions environment_options_;
  LiteRtOptions options_;
  const std::string graph_name_;
  LiteRtDispatchDeviceContext device_context_;
  const bool async_dispatch_;  // Indicates whether the Dispatch API can be
                               // invoked asynchronously.

  LiteRtExternalLiteRtBufferContextT* buffer_context_ = nullptr;
  std::vector<TfLiteOpaqueNode*> nodes_;
  std::vector<LiteRtDispatchInvocationContext> node_invocation_contexts_;

  // Store tensor IDs - get tensors on demand to avoid stale pointers
  std::vector<int> input_tensor_ids_;
  std::vector<int> output_tensor_ids_;
  std::vector<int> internal_tensor_ids_;

  std::unordered_map<int, LiteRtTensorBufferHandle> tensor_idx_to_handle_;  // NOLINT

  struct TensorInfo {
    LiteRtTensorBufferPtr tensor_buffer;
    LiteRtTensorBufferHandle buffer_handle;
    bool maybe_sync_with_cpu = false;
    size_t tensor_buffer_used_size = 0;
    bool attached = false;

    TensorInfo() = default;

    void MarkAsMaybeSyncWithCpu(size_t used_size) {
      maybe_sync_with_cpu = true;
      tensor_buffer_used_size = used_size;
    }
  };

  absl::node_hash_map<TfLiteOpaqueTensor*, TensorInfo> tensor_buffer_infos_;

  struct PortConnection {
    int node_idx;
    int port_idx;        // The index of the I/O node port.
    bool is_input_port;  // Wheter this connection is to a node input or to a
                         // node output.
  };

  absl::flat_hash_map<TfLiteOpaqueTensor*, std::vector<PortConnection>>
      io_tensors_port_connections_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_
