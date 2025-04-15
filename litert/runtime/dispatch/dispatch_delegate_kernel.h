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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_dispatch_delegate.h"
#include "litert/c/litert_metrics.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/runtime/metrics.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils/simple_opaque_delegate.h"

namespace litert::internal {

class ExternalLiteRtBufferContext;

// A TFL kernel that the interpreter calls to dispatch execution through the
// Dispatch API.
class DispatchDelegateKernel
    : public tflite::SimpleOpaqueDelegateKernelInterface {
 public:
  using Ptr = std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>;

  ~DispatchDelegateKernel() override;

  static Expected<Ptr> Create(std::string&& graph_name,
                              const LiteRtDispatchDelegateOptions& options,
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
  DispatchDelegateKernel(const LiteRtDispatchDelegateOptions& options,
                         std::string&& graph_name,
                         LiteRtDispatchDeviceContext device_context,
                         bool async_dispatch)
      : options_(options),
        graph_name_(std::move(graph_name)),
        device_context_(std::move(device_context)),
        async_dispatch_(async_dispatch) {}

  static Expected<ExternalLiteRtBufferContext*> GetBufferContext(
      TfLiteOpaqueContext* context);
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

  Expected<TensorBufferRequirements> GetBufferRequirements(
      int node_idx, TfLiteOpaqueTensor* io_tfl_tensor, int io_tensor_index,
      bool is_input) const;

  Expected<void> ComputeRequirements(TfLiteOpaqueContext* context);
  Expected<void> ComputeTensorPortConnections(TfLiteOpaqueContext* context);

  Expected<void> AllocateTensorBuffersIfNeeded();
  Expected<TensorBuffer> AllocateTensorBuffer(TfLiteOpaqueTensor* tfl_tensor);
  Expected<void> RegisterBufferWithDispatchApi(TfLiteOpaqueTensor* tfl_tensor,
                                               TensorBuffer&& tensor_buffer);

  Expected<void> AttachBuffersToInvocationContextsIfNeeded(
      TfLiteOpaqueContext* context);

  Expected<void> ScheduleAsyncExecution(TfLiteOpaqueContext* context);
  Expected<void> ScheduleSyncExecution(TfLiteOpaqueContext* context);

  const LiteRtDispatchDelegateOptions& options_;
  const std::string graph_name_;
  LiteRtDispatchDeviceContext device_context_;
  const bool async_dispatch_;  // Indicates whether the Dispatch API can be
                               // invoked asynchronously.

  ExternalLiteRtBufferContext* buffer_context_ = nullptr;
  std::vector<TfLiteOpaqueNode*> nodes_;
  std::vector<LiteRtDispatchInvocationContext> node_invocation_contexts_;
  std::vector<TfLiteOpaqueTensor*> input_tensors_;
  std::vector<TfLiteOpaqueTensor*> output_tensors_;
  std::vector<TfLiteOpaqueTensor*> internal_tensors_;

  struct TensorInfo {
    TensorBuffer tensor_buffer;
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
