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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <optional>

#include "neuron/api/NeuronAdapter.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/common/vendor_dispatch_base.h"
#include "litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

class LiteRtDispatchInvocationContextT
    : public litert::vendors::VendorInvocationContext {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<Ptr> Create(
      litert::mediatek::NeuronAdapterApi& neuron_adapter_api,
      LiteRtDispatchDeviceContextT& device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs);

  ~LiteRtDispatchInvocationContextT();

  // Override base class methods
  LiteRtStatus AttachInput(
      int graph_input_index,
      LiteRtTensorBufferHandle tensor_buffer_handle) override;
  LiteRtStatus AttachOutput(
      int graph_output_index,
      LiteRtTensorBufferHandle tensor_buffer_handle) override;
  LiteRtStatus DetachInput(
      int graph_input_index,
      LiteRtTensorBufferHandle tensor_buffer_handle) override;
  LiteRtStatus DetachOutput(
      int graph_output_index,
      LiteRtTensorBufferHandle tensor_buffer_handle) override;
  LiteRtStatus Invoke() override;
  void* GetBackendContext() override { return execution_; }

  // MediaTek-specific methods
  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

 private:
  class IoRequirementsBuilder {
   public:
    IoRequirementsBuilder(size_t buffer_size,
                          const std::vector<uint32_t>& padded_dimensions);
    litert::Expected<LiteRtTensorBufferRequirements> Create();

   private:
    size_t buffer_size_;
    std::vector<uint32_t> strides_;
  };

  LiteRtDispatchInvocationContextT(
      const litert::mediatek::NeuronAdapterApi& neuron_adapter_api,
      LiteRtDispatchDeviceContextT& device_context, NeuronModel* model,
      NeuronCompilation* compilation, NeuronExecution* execution,
      int num_inputs, int num_outputs)
      : neuron_adapter_api_(neuron_adapter_api),
        device_context_(device_context),
        model_(model),
        compilation_(compilation),
        execution_(execution),
        input_requirements_builders_(num_inputs),
        output_requirements_builders_(num_outputs) {}

  const litert::mediatek::NeuronAdapterApi& neuron_adapter_api_;
  LiteRtDispatchDeviceContextT& device_context_;
  NeuronModel* model_;
  NeuronCompilation* compilation_;
  NeuronExecution* execution_;
  std::vector<std::unique_ptr<IoRequirementsBuilder>>
      input_requirements_builders_;
  std::vector<std::unique_ptr<IoRequirementsBuilder>>
      output_requirements_builders_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
