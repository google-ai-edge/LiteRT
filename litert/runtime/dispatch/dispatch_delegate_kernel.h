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

#include "litert/c/litert_common.h"
#include "litert/c/litert_dispatch_delegate.h"
#include "litert/c/litert_metrics.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/c/c_api_types.h"  // from @org_tensorflow
#include "tensorflow/lite/c/common.h"  // from @org_tensorflow
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"  // from @org_tensorflow

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
                              const LiteRtDispatchDelegateOptions& options);

  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override;

  LiteRtStatus StartMetricsCollection(int detail_level);

  Expected<LiteRtMetricsT> StopMetricsCollection();

 private:
  DispatchDelegateKernel(const LiteRtDispatchDelegateOptions& options,
                         std::string&& graph_name,
                         LiteRtDispatchDeviceContext device_context,
                         bool async_dispatch)
      : options_(options),
        graph_name_(std::move(graph_name)),
        device_context_(device_context),
        async_dispatch_(async_dispatch) {}

  Expected<TensorBufferRequirements> GetBufferRequirements(
      const RankedTensorType& tensor_type, int io_tensor_index,
      bool is_input) const;

  // Creates a new tensor buffer for the given tensor. After that the created
  // tensor buffer is registered with RegisterLiteRtTensorBuffer().
  TfLiteStatus CreateAndSetBuffer(const TfLiteOpaqueTensor* tfl_opaque_tensor,
                                  int buffer_index, bool is_input);

  // Registers the given LiteRtTensorBuffer (and its size) with the Dispatch
  // API.
  // Also update the internal state (input_tensor_buffers_, etc.) to keep track
  // of the registered tensor buffers.
  TfLiteStatus RegisterLiteRtTensorBuffer(TensorBuffer&& tensor_buffer,
                                          size_t used_size, int buffer_index,
                                          bool is_input);

  // Registers LiteRtTensorBuffers for all inputs and outputs of the given
  // node.
  // Also update the internal state (input_tensor_buffers_, etc.) to keep track
  // of the registered tensor buffers.
  TfLiteStatus RegisterLiteRtTensorBuffers(TfLiteOpaqueContext* context,
                                           TfLiteOpaqueNode* node);

  const LiteRtDispatchDelegateOptions& options_;
  std::string graph_name_;
  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchInvocationContext invocation_context_ = nullptr;
  // Indicates whether the Dispatch API can be invoked asynchronously.
  const bool async_dispatch_;

  ExternalLiteRtBufferContext* buffer_context_ = nullptr;

  // Indicates whether the input tensor buffer requires a CPU sync before
  // invoking the Dispatch API.
  std::vector<bool> input_tensor_buffers_require_cpu_sync_;

  std::vector<TensorBuffer> input_tensor_buffers_;
  std::vector<LiteRtTensorBufferHandle> input_tensor_buffer_handles_;
  std::vector<size_t> input_tensor_buffer_used_size_;

  // Indicates whether the output tensor buffer requires a CPU sync after
  // invoking the Dispatch API.
  std::vector<bool> output_tensor_buffers_require_cpu_sync_;

  std::vector<TensorBuffer> output_tensor_buffers_;
  std::vector<LiteRtTensorBufferHandle> output_tensor_buffer_handles_;
  std::vector<size_t> output_tensor_buffer_used_size_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_
